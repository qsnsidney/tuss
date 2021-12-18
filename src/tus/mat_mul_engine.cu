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
#include "mat_mul_kernel.cuh"

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
        constexpr size_t src_index = 0;
        constexpr size_t dest_index = 1;

        data_t_3d* d_X[2] = {nullptr, nullptr};
        gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size));

        data_t_3d* d_A[2] = {nullptr, nullptr};
        gpuErrchk(cudaMalloc((void **)&d_A[src_index], vector_size));
        gpuErrchk(cudaMalloc((void **)&d_A[dest_index], vector_size));

        data_t_3d* d_V[2] = {nullptr, nullptr};
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

                calculate_acceleration<<<nblocks, block_size_>>>(nBody, d_X[dest_index], d_M, //input
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

        for(const auto i : {src_index, dest_index}){
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