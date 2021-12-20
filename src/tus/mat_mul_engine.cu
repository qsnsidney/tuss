#include "mat_mul_engine.cuh"
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
            data_t_3d displacement = (x_source - x_self);
            data_t denominator = power_norm(displacement);
            data_t_3d new_term = (displacement * mass[j] / denominator);
            accumulated_accer = accumulated_accer + new_term;
            //printf("tid = %d, new_term %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z);
        }
        acceleration[tid] = accumulated_accer;
    }
}

__global__ inline void calculate_field(unsigned nbody, unsigned target_ibody, data_t_3d *location, data_t *field)
{
    const unsigned source_ibody = threadIdx.x + blockDim.x * blockIdx.x;
    if (source_ibody < nbody)
    {
        const data_t_3d x_target = location[target_ibody];
        const data_t_3d x_source = location[source_ibody];
        const data_t_3d numerator = (x_source - x_target);
        const data_t denominator = power_norm(x_target, x_source);
        const data_t_3d source_field = numerator / denominator;

        field[source_ibody] = source_field.x;
        field[nbody + source_ibody] = source_field.y;
        field[nbody + nbody + source_ibody] = source_field.z;
    }
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
    MAT_MUL_ENGINE::MAT_MUL_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                                   CORE::DT dt,
                                   int block_size,
                                   std::optional<std::string> system_state_log_dir_opt) : ENGINE(std::move(system_state_ic), dt, std::move(system_state_log_dir_opt)),
                                                                                          block_size_(block_size)
    {
    }

    CORE::SYSTEM_STATE MAT_MUL_ENGINE::execute(int n_iter, CORE::TIMER &timer)
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
        data_t *d_M = nullptr;
        gpuErrchk(cudaMalloc((void **)&d_M, data_size));
        /*
     *   create double buffer on device side
     */
        unsigned src_index = 0;
        unsigned dest_index = 1;

        data_t_3d *d_X[2] = {nullptr, nullptr};
        gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size));

        data_t_3d *d_A[2] = {nullptr, nullptr};
        gpuErrchk(cudaMalloc((void **)&d_A[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_A[dest_index], vector_size));

        data_t_3d *d_V[2] = {nullptr, nullptr};
        gpuErrchk(cudaMalloc((void **)&d_V[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_V[dest_index], vector_size));

        data_t_3d *d_V_half = nullptr;
        gpuErrchk(cudaMalloc((void **)&d_V_half, vector_size));

        // d_Field[0..nBody] = field.x
        // d_Field[nBody..2*nBody] = field.y
        // d_Field[2*nBody..3*nBody] = field.z
        data_t *d_Field = nullptr;
        gpuErrchk(cudaMalloc((void**)&d_Field, sizeof(data_t) * nBody * 3));

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
        calculate_acceleration<<<nblocks, block_size_>>>(nBody, d_X[src_index], d_M, d_A[src_index]);
        timer.elapsed_previous("Calculated initial acceleration");

        {
            CORE::TIMER core_timer("all_iters");
            for (int i_iter = 0; i_iter < n_iter; i_iter++)
            {
                update_step_pos<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_X[src_index], d_V[src_index], d_A[src_index], d_M, //input
                                                            d_X[dest_index], d_V_half);                                               // output

                cudaDeviceSynchronize();

                // calculate_acceleration<<<nblocks, block_size_>>>(nBody, d_X[dest_index], d_M, //input
                //                                                    d_A[dest_index]);            // output
                // cudaDeviceSynchronize();

                for(size_t ibody = 0; ibody < nBody; ibody++) {
                    // prepare for field
                    // Since everytime, there is only one target body,
                    // can we directly pass the location of that into the kernel call?
                    calculate_field<<<nblocks, block_size_>>>(nBody, ibody, d_X[dest_index], // input
                        d_Field); // output
                    // Does not matter
                    // cudaDeviceSynchronize();
                }

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

                std::swap(src_index, dest_index);
            }
            cudaDeviceSynchronize();
        }

        // at end, the final data is actually at src_index because the last swap
        cudaMemcpy(h_output_X, d_X[src_index], vector_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_V, d_V[src_index], vector_size, cudaMemcpyDeviceToHost);
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
        cudaFree(d_Field);
        cudaDeviceReset();

        return system_state_result;
    }
}