#include "nvda_improved_engine.cuh"
#include "core/timer.h"

#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>

#include "core/physics.hpp"
#include "core/serde.h"
#include "helper.cuh"
#include "data_t.cuh"

__global__ inline void update_step_pos_f4(unsigned nbody, data_t step_size, float4 *i_location, data_t_3d *i_velocity, float4 *i_accer, // new accer is accer at i+1 iteration
                                          float4 *o_location, data_t_3d *velocity_half)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        // v1/2          =         vi      +     ai  *          1/2 *    dt
        velocity_half[tid].x = i_velocity[tid].x + i_accer[tid].x * ((data_t)0.5 * step_size);
        velocity_half[tid].y = i_velocity[tid].y + i_accer[tid].y * ((data_t)0.5 * step_size);
        velocity_half[tid].z = i_velocity[tid].z + i_accer[tid].z * ((data_t)0.5 * step_size);
        // Xi+1         =      xi         +       vi        *     dt    +    ai   *     1/2     *     (dt)^2
        o_location[tid].x = i_location[tid].x + i_velocity[tid].x * step_size + i_accer[tid].x * (data_t)0.5 * powf(step_size, 2);
        o_location[tid].y = i_location[tid].y + i_velocity[tid].y * step_size + i_accer[tid].y * (data_t)0.5 * powf(step_size, 2);
        o_location[tid].z = i_location[tid].z + i_velocity[tid].z * step_size + i_accer[tid].z * (data_t)0.5 * powf(step_size, 2);
        o_location[tid].w = i_location[tid].w;

        //printf("tid = %d, half_v %f, %f, %f\no_location %f, %f, %f\n", tid, velocity_half[tid].x, velocity_half[tid].y, velocity_half[tid].z, o_location[tid].x, o_location[tid].y, o_location[tid].z);
    }
}

__global__ inline void update_step_vel_f4(unsigned nbody, data_t step_size, float4 *new_accer, data_t_3d *velocity_half, // new accer is accer at i+1 iteration
                                          data_t_3d *o_velocity)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        o_velocity[tid].x = velocity_half[tid].x + new_accer[tid].x * ((data_t)0.5 * step_size);
        o_velocity[tid].y = velocity_half[tid].y + new_accer[tid].y * ((data_t)0.5 * step_size);
        o_velocity[tid].z = velocity_half[tid].z + new_accer[tid].z * ((data_t)0.5 * step_size);
        //printf("tid = %d, update_v %f, %f, %f\n", tid, o_velocity[tid].x, o_velocity[tid].y, o_velocity[tid].z);
    }
}

/*
 * Functions below are taken from https://www.researchgate.net/publication/291770155_Fast_N-body_simulation_with_CUDA with
 * only necessary modifications such as boundary condition check and parameter fixing.
 * The code is purely intended to be used as a reference for performance comparsion.
 *
 */

__device__ inline float3
AccumulatebodyBodyInteraction_improved(float4 bi, float4 bj, float3 ai)
{
    float3 r;
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + CORE::UNIVERSE::epislon_square;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ inline float3
tile_calculation_improved(float4 myPosition, float3 accel, int accum_length)
{
    int i;
    extern __shared__ float4 shPosition[];
    for (i = 0; i < accum_length; i += 4)
    {
        // we don't need to check the object index.
        // because the vector subtration of oneself will just yields 0.
        // hence contributes no acceleration.
        accel = AccumulatebodyBodyInteraction_improved(myPosition, shPosition[i], accel);
        accel = AccumulatebodyBodyInteraction_improved(myPosition, shPosition[i + 1], accel);
        accel = AccumulatebodyBodyInteraction_improved(myPosition, shPosition[i + 2], accel);
        accel = AccumulatebodyBodyInteraction_improved(myPosition, shPosition[i + 3], accel);
    }
    return accel;
}

// each calculate forces handles one body
__global__ inline void
calculate_forces_improved(int N, void *devX, void *devA, int p)
{
    extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)devX;
    float4 *globalA = (float4 *)devA;
    float4 myPosition;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // we don't skip the object even if it's gtid > N.
    // reasons explained later.
    myPosition = globalX[gtid];
    for (i = 0, tile = 0; i < N; i += blockDim.x, tile++)
    {

        // decide which piece of memory to read into the shared mem
        int idx = tile * blockDim.x + threadIdx.x;

        // printf("gid: %d. idx: %d\n",gtid, idx);
        // It is possible that the current mem to read is out of bound
        // but the thread itself is dealing with a valid body
        // for example, when there are 48 bodies with block_size = 32.
        // in the 2nd iteration, thread of body 24 will try to read sharemem
        // of body 56. but we should not skip body 24's accleration accumulatio
        //if(idx >= N) {
        //    shPosition[threadIdx.x] = {0.0f, 0.0f, 0.0f, 0.0f};
        //}
        //else {
        shPosition[threadIdx.x] = globalX[idx];
        //

        // we have to skip the thread that's greater than gtid here
        // instead of earlier, because the thread could be reading some
        // shared mem data. imagine a case of block size = 8 and 9 body.
        // then the thread with gtid 9 will be reading the body1's location
        // in the first iteration. now the thread is done with loading the shared mem
        // so we can skip it.

        // Ideally, we should take care of the case where the last tile contains less than
        // num_block of data. only let the tiled function process min(blocksize, remaining elements)
        // in length. but because we already load out of bound shared mem with 0s. we don't have to
        // worry about out of bound anymore.
        __syncthreads();
        acc = tile_calculation_improved(myPosition, acc, blockDim.x);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
}

namespace
{
    CORE::SYSTEM_STATE generate_system_state(const float4 *h_X, const data_t_3d *h_V, const size_t nbody)
    {
        CORE::SYSTEM_STATE system_state;
        system_state.reserve(nbody);
        for (size_t i_body = 0; i_body < nbody; i_body++)
        {
            CORE::POS pos_temp{h_X[i_body].x, h_X[i_body].y, h_X[i_body].z};
            CORE::VEL vel_temp{h_V[i_body].x, h_V[i_body].y, h_V[i_body].z};
            system_state.emplace_back(pos_temp, vel_temp, h_X[i_body].w);
        }
        return system_state;
    }
}

namespace TUS
{
    NVDA_IMPROVED_ENGINE::NVDA_IMPROVED_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                                               CORE::DT dt,
                                               int block_size,
                                               std::optional<std::string> system_state_log_dir_opt) : ENGINE(std::move(system_state_ic), dt, std::move(system_state_log_dir_opt)),
                                                                                                      block_size_(block_size)
    {
    }

    CORE::SYSTEM_STATE NVDA_IMPROVED_ENGINE::execute(int n_iter, CORE::TIMER &timer)
    {
        size_t nBody = system_state_snapshot().size();

        /* BIN file of initial conditions */
        const auto &ic = system_state_snapshot();

        // random initializer just for now
        size_t vector_size_3d = sizeof(data_t_3d) * nBody;
        size_t vector_size_4d = sizeof(float4) * nBody;
        size_t vector_size_4dx = sizeof(float4) * ((nBody + (block_size_ - 1)) / block_size_) * block_size_;
        /*
     *   host side memory allocation
     */
        data_t_3d *h_V, *h_output_V;
        float4 *h_X, *h_A, *h_output_X;

        host_malloc_helper((void **)&h_V, vector_size_3d);
        host_malloc_helper((void **)&h_output_V, vector_size_3d);

        host_malloc_helper((void **)&h_X, vector_size_4dx);
        host_malloc_helper((void **)&h_A, vector_size_4d);
        host_malloc_helper((void **)&h_output_X, vector_size_4d);

        timer.elapsed_previous("allocated host side memory");
        /*
     *   input randome initialize
     */

        for (int i = 0; i < ((nBody + (block_size_ - 1)) / block_size_) * block_size_; i++)
        {
            h_X[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        parse_ic_f4(h_X, h_V, ic);
        timer.elapsed_previous("deserialize_body_state_vec_from_csv");

        /*
         * create double buffer on device side
         */
        float4 **d_X, **d_A;
        unsigned src_index = 0;
        unsigned dest_index = 1;
        d_X = (float4 **)malloc(2 * sizeof(float4 *));
        gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size_4dx));
        gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size_4dx));

        d_A = (float4 **)malloc(2 * sizeof(float4 *));
        gpuErrchk(cudaMalloc((void **)&d_A[src_index], vector_size_4d));
        gpuErrchk(cudaMalloc((void **)&d_A[dest_index], vector_size_4d));

        data_t_3d **d_V;
        d_V = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
        gpuErrchk(cudaMalloc((void **)&d_V[src_index], vector_size_3d));
        gpuErrchk(cudaMalloc((void **)&d_V[dest_index], vector_size_3d));

        data_t_3d *d_V_half;
        gpuErrchk(cudaMalloc((void **)&d_V_half, vector_size_3d));

        timer.elapsed_previous("allocated device memory");
        /*
         *   create double buffer on device side
         */
        // cudaMemcpy(d_A[0], h_A, vector_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_X[src_index], h_X, vector_size_4dx, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V[src_index], h_V, vector_size_3d, cudaMemcpyHostToDevice);
        timer.elapsed_previous("copied input data from host to device");

        // nthread is assigned to either 32 by default or set to a custom power of 2 by user
        std::cout << "Set thread_per_block to " << block_size_ << std::endl;
        unsigned nblocks = (nBody + block_size_ - 1) / block_size_;

        // calculate the initialia acceleration
        calculate_forces_improved<<<nblocks, block_size_, block_size_ * sizeof(float4)>>>(nBody, d_X[src_index], d_A[src_index], block_size_);
        timer.elapsed_previous("Calculated initial acceleration");

        {
            CORE::TIMER core_timer("all_iters");
            for (int i_iter = 0; i_iter < n_iter; i_iter++)
            {
                update_step_pos_f4<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_X[src_index], d_V[src_index], d_A[src_index], //input
                                                               d_X[dest_index], d_V_half);                                          // output

                cudaDeviceSynchronize();

                calculate_forces_improved<<<nblocks, block_size_, block_size_ * sizeof(float4)>>>(nBody, d_X[dest_index],        //input
                                                                                                    d_A[dest_index], block_size_); // output

                cudaDeviceSynchronize();

                update_step_vel_f4<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_A[dest_index], d_V_half, //input
                                                               d_V[dest_index]);                               // output
                cudaDeviceSynchronize();

                timer.elapsed_previous(std::string("iter") + std::to_string(i_iter), CORE::TIMER::TRIGGER_LEVEL::INFO);

                if (is_system_state_logging_enabled())
                {
                    cudaMemcpy(h_output_X, d_X[dest_index], vector_size_4d, cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_output_V, d_V[dest_index], vector_size_4d, cudaMemcpyDeviceToHost);

                    if (i_iter == 0)
                    {
                        push_system_state_to_log(generate_system_state(h_X, h_V, nBody));
                    }
                    push_system_state_to_log(generate_system_state(h_output_X, h_output_V, nBody));

                    if (i_iter % 10 == 0)
                    {
                        serialize_system_state_log();
                    }

                    timer.elapsed_previous(std::string("Transfer to CPU"), CORE::TIMER::TRIGGER_LEVEL::INFO);
                }

                swap(src_index, dest_index);
            }
            cudaDeviceSynchronize();
        }

        // at end, the final data is actually at src_index because the last swap
        cudaMemcpy(h_output_X, d_X[src_index], vector_size_4d, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_V, d_V[src_index], vector_size_3d, cudaMemcpyDeviceToHost);

        // Hack Hack Hack. dump out the data
        cudaMemcpy(h_A, d_A[src_index], vector_size_4d, cudaMemcpyDeviceToHost);

        write_debug_output(name(), h_output_X, h_output_V, h_A, nBody);

        timer.elapsed_previous("copied output back to host");

        //Just for debug purpose on small inputs
        // for (unsigned i = 0; i < nBody; i++)
        // {
        //    printf("object = %d, %f, %f, %f\n", i, h_output_X[i].x, h_output_X[i].y, h_output_X[i].z);
        // }

        auto system_state_result = generate_system_state(h_output_X, h_output_V, nBody);

        cudaFreeHost(h_X);
        cudaFreeHost(h_A);
        cudaFreeHost(h_V);
        cudaFreeHost(h_output_X);
        cudaFreeHost(h_output_V);

        for (const auto i : {src_index, dest_index})
        {
            cudaFree(d_X[i]);
            cudaFree(d_V[i]);
            cudaFree(d_A[i]);
        }
        cudaFree(d_V_half);
        cudaDeviceReset();

        return system_state_result;
    }
}