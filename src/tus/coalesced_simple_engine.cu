#include "coalesced_simple_engine.cuh"
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

__global__ inline void calculate_acceleration_f4(unsigned nbody, float4 *location, float4 *acceleration)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        data_t_3d accumulated_accer = make_data_t_3d(0, 0, 0);
        data_t_3d x_self = make_data_t_3d(location[tid].x, location[tid].y, location[tid].z);
        for (unsigned j = 0; j < nbody; j++)
        {
            if (j == tid)
            {
                continue;
            }
            // source of gravitiy
            data_t_3d x_source = make_float3(location[j].x, location[j].y, location[j].z);
            data_t mass = location[j].w;

            data_t_3d numerator = (x_source - x_self) * mass;
            data_t denominator = power_norm(x_self, x_source);
            data_t_3d new_term = (numerator / denominator);
            accumulated_accer = accumulated_accer + new_term;
            //printf("tid = %d, new_term %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z);
        }
        acceleration[tid] = make_float4(accumulated_accer.x, accumulated_accer.y, accumulated_accer.z, 0);
    }
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
    COALESCED_SIMPLE_ENGINE::COALESCED_SIMPLE_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                                                     CORE::DT dt,
                                                     int block_size,
                                                     std::optional<std::string> system_state_log_dir_opt) : ENGINE(std::move(system_state_ic), dt, std::move(system_state_log_dir_opt)),
                                                                                                            block_size_(block_size)
    {
    }

    CORE::SYSTEM_STATE COALESCED_SIMPLE_ENGINE::execute(int n_iter, CORE::TIMER &timer)
    {
        size_t nBody = system_state_snapshot().size();

        /* BIN file of initial conditions */
        const auto &ic = system_state_snapshot();

        // random initializer just for now
        size_t vector_size_3d = sizeof(data_t_3d) * nBody;
        size_t vector_size_4d = sizeof(float4) * nBody;

        /*
     *   host side memory allocation
     */
        data_t_3d *h_V, *h_output_V;
        float4 *h_X, *h_A, *h_output_X;

        host_malloc_helper((void **)&h_V, vector_size_3d);
        host_malloc_helper((void **)&h_output_V, vector_size_3d);

        host_malloc_helper((void **)&h_X, vector_size_4d);
        host_malloc_helper((void **)&h_A, vector_size_4d);
        host_malloc_helper((void **)&h_output_X, vector_size_4d);

        timer.elapsed_previous("allocated host side memory");
        /*
     *   input randome initialize
     */

        parse_ic_f4(h_X, h_V, ic);
        timer.elapsed_previous("deserialize_body_state_vec_from_csv");

        /*
         * create double buffer on device side
         */
        float4 **d_X, **d_A;
        unsigned src_index = 0;
        unsigned dest_index = 1;
        d_X = (float4 **)malloc(2 * sizeof(float4 *));
        gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size_4d));
        gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size_4d));

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
        cudaMemcpy(d_X[src_index], h_X, vector_size_4d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V[src_index], h_V, vector_size_3d, cudaMemcpyHostToDevice);
        timer.elapsed_previous("copied input data from host to device");

        // nthread is assigned to either 32 by default or set to a custom power of 2 by user
        std::cout << "Set thread_per_block to " << block_size_ << std::endl;
        unsigned nblocks = (nBody + block_size_ - 1) / block_size_;

        // calculate the initialia acceleration
        calculate_acceleration_f4<<<nblocks, block_size_>>>(nBody, d_X[src_index], d_A[src_index]);
        timer.elapsed_previous("Calculated initial acceleration");

        {
            CORE::TIMER core_timer("all_iters");
            for (int i_iter = 0; i_iter < n_iter; i_iter++)
            {
                update_step_pos_f4<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_X[src_index], d_V[src_index], d_A[src_index], //input
                                                               d_X[dest_index], d_V_half);                                          // output

                cudaDeviceSynchronize();

                calculate_acceleration_f4<<<nblocks, block_size_>>>(nBody, d_X[dest_index], //input
                                                                      d_A[dest_index]);       // output

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