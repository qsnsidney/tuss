#include "tiled_simple_engine.cuh"
#include "core/timer.h"

#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include <fstream>

#include "core/physics.hpp"
#include "core/serde.h"
#include "helper.cuh"
#include "data_t.cuh"

__global__ inline void update_step_pos(unsigned nbody, data_t step_size, data_t_3d *i_location, data_t_3d *i_velocity, data_t_3d *i_accer, data_t *mass, // new accer is accer at i+1 iteration
                                       data_t_3d *o_location, data_t_3d *velocity_half)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        // v1/2          =         vi      +     ai  *          1/2 *    dt
        velocity_half[tid] = i_velocity[tid] + i_accer[tid] * ((data_t)0.5 * step_size);
        // Xi+1         =      xi         +       vi        *     dt    +    ai   *     1/2     *     (dt)^2
        o_location[tid] = i_location[tid] + i_velocity[tid] * step_size + i_accer[tid] * (data_t)0.5 * powf(step_size, 2);

        //printf("tid = %d, half_v %f, %f, %f\no_location %f, %f, %f\n", tid, velocity_half[tid].x, velocity_half[tid].y, velocity_half[tid].z, o_location[tid].x, o_location[tid].y, o_location[tid].z);
    }
}

__global__ inline void update_step_vel(unsigned nbody, data_t step_size, data_t *mass, data_t_3d *new_accer, data_t_3d *velocity_half, // new accer is accer at i+1 iteration
                                       data_t_3d *o_velocity)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        o_velocity[tid] = velocity_half[tid] + new_accer[tid] * ((data_t)0.5 * step_size);
        //printf("tid = %d, update_v %f, %f, %f\n", tid, o_velocity[tid].x, o_velocity[tid].y, o_velocity[tid].z);
    }
}

__global__ inline void calculate_acceleration_faster(unsigned nbody, data_t_3d *location, data_t *mass, data_t_3d *acceleration)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        data_t_3d accumulated_accer = make_data_t_3d(0, 0, 0);
        data_t_3d x_self = location[tid];
        for (unsigned j = 0; j < nbody; j++)
        {
            if (j == tid)
            {
                continue;
            }
            // source of gravitiy
            data_t_3d x_source = location[j];
            data_t_3d numerator = (x_source - x_self);
            data_t denominator_inv = power_norm_inverse(x_self, x_source);
            data_t coefficient = denominator_inv * mass[j];
            data_t_3d new_term = numerator * coefficient;
            accumulated_accer = accumulated_accer + new_term;
            //printf("tid = %d, new_term %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z);
        }
        acceleration[tid] = accumulated_accer;
    }
}

__global__ inline void calculate_acceleration(unsigned nbody, data_t_3d *location, data_t *mass, data_t_3d *acceleration)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        data_t_3d accumulated_accer = make_data_t_3d(0, 0, 0);
        data_t_3d x_self = location[tid];
        for (unsigned j = 0; j < nbody; j++)
        {
            if (j == tid)
            {
                continue;
            }
            // source of gravitiy
            data_t_3d x_source = location[j];
            data_t_3d numerator = (x_source - x_self) * mass[j];
            data_t denominator = power_norm(x_self, x_source);
            data_t_3d new_term = (numerator / denominator);
            accumulated_accer = accumulated_accer + new_term;
            //printf("tid = %d, new_term %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z);
        }
        acceleration[tid] = accumulated_accer;
    }
}

__device__ inline float3
AccumulatebodyBodyInteraction(float3 bi, float3 bj, float3 ai, float massj)
{
    float3 r;
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z + CORE::UNIVERSE::epislon_square);
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    // s = m_j * invDistCube [1 FLOP]
    float s = massj / distSixth;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ inline float3
tile_calculation(float3 myPosition, float3 accel, int accum_length, int mass_offset, int offset)
{
    int i;
    extern __shared__ float shPosition[];
    for (i = 0; i < accum_length; i++)
    {
        // we don't need to check the object index.
        // because the vector subtration of oneself will just yields 0.
        // hence contributes no acceleration.
        float3 sharedp = make_float3(shPosition[i * 3], shPosition[i * 3 + 1], shPosition[i * 3 + 2]);
        accel = AccumulatebodyBodyInteraction(myPosition, sharedp, accel, shPosition[mass_offset + i]);
    }
    return accel;
}

// each calculate forces handles one body
__global__ inline void
calculate_forces(int N, void *devX, float *devM, void *devA)
{
    extern __shared__ float shPosition[];
    float3 *globalX = (float3 *)devX;
    float3 *globalA = (float3 *)devA;
    float3 myPosition;
    int mass_offset = 3 * blockDim.x;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // we don't skip the object even if it's gtid > N.
    // reasons explained later.
    if (gtid < N)
    {
        myPosition = globalX[gtid];
    }
    else
    {
        myPosition = {0.0f, 0.0f, 0.0f};
    }
    for (i = 0, tile = 0; i < N; i += blockDim.x, tile++)
    {

        // decide which piece of memory to read into the shared mem
        int idx = tile * blockDim.x + threadIdx.x;

        // printf("gid: %d. idx: %d\n",gtid, idx);
        // It is possible that the current mem to read is out of bound
        // but the thread itself is dealing with a valid body
        // for example, when there are 48 bodies with block_size = 32.
        // in the 2nd iteration, thread of body 24 will try to read sharemem
        // of body 56. but we should not skip body 24's accleration accumulation
        if (idx >= N)
        {
            shPosition[3 * threadIdx.x] = 0.0f;
            shPosition[3 * threadIdx.x + 1] = 0.0f;
            shPosition[3 * threadIdx.x + 2] = 0.0f;
            shPosition[mass_offset + threadIdx.x] = 0.0f;
        }
        else
        {
            shPosition[3 * threadIdx.x] = globalX[idx].x;
            shPosition[3 * threadIdx.x + 1] = globalX[idx].y;
            shPosition[3 * threadIdx.x + 2] = globalX[idx].z;
            shPosition[mass_offset + threadIdx.x] = devM[idx];
        }

        // we have to skip the thread that's greater than gtid here
        // instead of earlier, because the thread could be reading some
        // shared mem data. imagine a case of block size = 8 and 9 body.
        // then the thread with gtid 9 will be reading the body1's location
        // in the first iteration. now the thread is done with loading the shared mem
        // so we can skip it.
        if (gtid >= N)
        {
            continue;
        }
        auto offset = tile * blockDim.x;
        // Ideally, we should take care of the case where the last tile contains less than
        // num_block of data. only let the tiled function process min(blocksize, remaining elements)
        // in length. but because we already load out of bound shared mem with 0s. we don't have to
        // worry about out of bound anymore.
        __syncthreads();
        acc = tile_calculation(myPosition, acc, blockDim.x, mass_offset, offset);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float3 acc3 = {acc.x, acc.y, acc.z};
    globalA[gtid] = acc3;
}

namespace
{
    CORE::SYSTEM_STATE generate_system_state(const data_t_3d *h_X, const data_t_3d *h_V, const data_t *mass, const size_t nbody)
    {
        CORE::SYSTEM_STATE system_state;
        system_state.reserve(nbody);
        for (size_t i_body = 0; i_body < nbody; i_body++)
        {
            CORE::POS pos_temp{h_X[i_body].x, h_X[i_body].y, h_X[i_body].z};
            CORE::VEL vel_temp{h_V[i_body].x, h_V[i_body].y, h_V[i_body].z};
            system_state.emplace_back(pos_temp, vel_temp, mass[i_body]);
        }
        return system_state;
    }
}

namespace TUS
{
    TILED_SIMPLE_ENGINE::TILED_SIMPLE_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                                             CORE::DT dt,
                                             int block_size,
                                             std::optional<std::string> system_state_log_dir_opt) : ENGINE(std::move(system_state_ic), dt, std::move(system_state_log_dir_opt)),
                                                                                                    block_size_(block_size)
    {
    }

    CORE::SYSTEM_STATE TILED_SIMPLE_ENGINE::execute(int n_iter, CORE::TIMER &timer)
    {
        size_t nBody = system_state_snapshot().size();

        /* BIN file of initial conditions */
        const auto &ic = system_state_snapshot();

        // random initializer just for now
        size_t vector_size = sizeof(data_t_3d) * nBody;
        size_t data_size = sizeof(data_t) * nBody;

        /*
     *   host side memory allocation
     */
        data_t_3d *h_X, *h_A, *h_V, *h_output_X, *h_output_V;
        data_t *h_M;
        host_malloc_helper((void **)&h_X, vector_size);
        host_malloc_helper((void **)&h_A, vector_size);
        host_malloc_helper((void **)&h_V, vector_size);
        host_malloc_helper((void **)&h_output_X, vector_size);
        host_malloc_helper((void **)&h_output_V, vector_size);
        host_malloc_helper((void **)&h_M, data_size);
        timer.elapsed_previous("allocated host side memory");
        /*
     *   input randome initialize
     */

        parse_ic(h_X, h_V, h_M, ic);
        timer.elapsed_previous("deserialize_system_state_from_csv");

        /*
     *  mass 
     */
        data_t *d_M;
        gpuErrchk(cudaMalloc((void **)&d_M, data_size));
        /*
     *   create double buffer on device side
     */
        data_t_3d **d_X, **d_A, **d_V;
        unsigned src_index = 0;
        unsigned dest_index = 1;
        d_X = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
        gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size));

        d_A = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
        gpuErrchk(cudaMalloc((void **)&d_A[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_A[dest_index], vector_size));

        d_V = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
        gpuErrchk(cudaMalloc((void **)&d_V[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_V[dest_index], vector_size));

        data_t_3d *d_V_half;
        gpuErrchk(cudaMalloc((void **)&d_V_half, vector_size));

        timer.elapsed_previous("allocated device memory");
        /*
     *   create double buffer on device side
     */
        // cudaMemcpy(d_A[0], h_A, vector_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_X[src_index], h_X, vector_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V[src_index], h_V, vector_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_M, h_M, data_size, cudaMemcpyHostToDevice);
        timer.elapsed_previous("copied input data from host to device");

        // nthread is assigned to either 32 by default or set to a custom power of 2 by user
        std::cout << "Set thread_per_block to " << block_size_ << std::endl;
        unsigned nblocks = (nBody + block_size_ - 1) / block_size_;

        // calculate the initialia acceleration

        calculate_forces<<<nblocks, block_size_, block_size_ * sizeof(float4)>>>(nBody, d_X[src_index], d_M, d_A[src_index]);
        timer.elapsed_previous("Calculated initial acceleration");

        {
            CORE::TIMER core_timer("all_iters");
            for (int i_iter = 0; i_iter < n_iter; i_iter++)
            {
                update_step_pos<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_X[src_index], d_V[src_index], d_A[src_index], d_M, //input
                                                            d_X[dest_index], d_V_half);                                               // output

                cudaDeviceSynchronize();

                calculate_forces<<<nblocks, block_size_, block_size_ * sizeof(float4)>>>(nBody, d_X[dest_index], d_M, //input
                                                                                           d_A[dest_index]);            // output

                cudaDeviceSynchronize();

                update_step_vel<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_M, d_A[dest_index], d_V_half, //input
                                                            d_V[dest_index]);                                    // output
                cudaDeviceSynchronize();

                timer.elapsed_previous(std::string("iter") + std::to_string(i_iter), CORE::TIMER::TRIGGER_LEVEL::INFO);

                if (is_system_state_logging_enabled())
                {
                    cudaMemcpy(h_output_X, d_X[dest_index], vector_size, cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_output_V, d_V[dest_index], vector_size, cudaMemcpyDeviceToHost);

                    if (i_iter == 0)
                    {
                        push_system_state_to_log(generate_system_state(h_X, h_V, h_M, nBody));
                    }
                    push_system_state_to_log(generate_system_state(h_output_X, h_output_V, h_M, nBody));

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
        // at end, the final data is actually at src_index because the last swap
        cudaMemcpy(h_output_X, d_X[src_index], vector_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_V, d_V[src_index], vector_size, cudaMemcpyDeviceToHost);

#if 0
        // Hack Hack Hack. dump out the data
        cudaMemcpy(h_A, d_A[src_index], vector_size, cudaMemcpyDeviceToHost);

        std::ofstream X_file;
        std::ofstream V_file;
        std::ofstream A_file;
        X_file.open ("TiledX.output");
        V_file.open ("TiledV.output");
        A_file.open ("TiledA.output");
        for(int i = 0; i < nBody; i++) {
            X_file << h_output_X[i].x << "\n";
            X_file << h_output_X[i].y << "\n";
            X_file << h_output_X[i].z << "\n";
            V_file << h_output_V[i].x << "\n";
            V_file << h_output_V[i].y << "\n";
            V_file << h_output_V[i].z << "\n";
            A_file << h_A[i].x << "\n";
            A_file << h_A[i].y << "\n";
            A_file << h_A[i].z << "\n";
        }
        X_file.close();
        V_file.close();
        A_file.close();
#endif
        timer.elapsed_previous("copied output back to host");

        // Just for debug purpose on small inputs
        //for (unsigned i = 0; i < nBody; i++)
        //{
        //   printf("object = %d, %f, %f, %f\n", i, h_output_X[i].x, h_output_X[i].y, h_output_X[i].z);
        //}

        auto system_state_result = generate_system_state(h_output_X, h_output_V, h_M, nBody);

        cudaFreeHost(h_X);
        cudaFreeHost(h_A);
        cudaFreeHost(h_V);
        cudaFreeHost(h_output_X);
        cudaFreeHost(h_output_V);
        cudaFreeHost(h_M);

        for (const auto i : {src_index, dest_index})
        {
            cudaFree(d_X[i]);
            cudaFree(d_V[i]);
            cudaFree(d_A[i]);
        }
        cudaFree(d_V_half);
        cudaFree(d_M);
        cudaDeviceReset();

        return system_state_result;
    }
}