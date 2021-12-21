#include "small_N_engine.cuh"
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
#include "small_N_kernel.cuh"

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
    SMALL_N_ENGINE::SMALL_N_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                                       CORE::DT dt,
                                       int block_size,
                                       int tb_len,
                                       int tb_wid,
                                       int unroll_factor,
                                       int tpb,
                                       std::optional<std::string> system_state_log_dir_opt) : ENGINE(std::move(system_state_ic), dt, std::move(system_state_log_dir_opt)),
                                                                                              block_size_(block_size), tb_len_(tb_len), tb_wid_(tb_wid), 
                                                                                              unroll_factor_(unroll_factor), tpb_(tpb)
    {
    }

    CORE::SYSTEM_STATE SMALL_N_ENGINE::execute(int n_iter, CORE::TIMER &timer)
    {
        size_t nBody = system_state_snapshot().size();

        /* BIN file of initial conditions */
        const auto &ic = system_state_snapshot();

        // random initializer just for now
        size_t vector_size_3d = sizeof(data_t_3d) * nBody;
        size_t vector_size_4d = sizeof(float4) * nBody;
        size_t vector_size_4dx = sizeof(float4) * ((nBody + (block_size_ - 1))/block_size_) * block_size_;
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

        for(int i = 0; i < ((nBody + (block_size_ - 1))/block_size_) * block_size_; i++) {
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

        float4 * d_intermidiate_A;
        int summation_result_per_body = (nBody + unroll_factor_ - 1) / unroll_factor_;
        std::cout << "summation result per body is " << summation_result_per_body << std::endl;
        gpuErrchk(cudaMalloc((void **)&d_intermidiate_A, sizeof(float4) * nBody * summation_result_per_body));

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


        std::cout << "set number of body to " << nBody << std::endl;
        dim3 block( tb_len_, tb_wid_ );
        int column_per_block = (tb_len_ * unroll_factor_);
        dim3 grid( (nBody + column_per_block-1)/column_per_block, (nBody + block.y-1)/block.y );

        std::cout << "2d block dimension: (" << tb_len_ << "," << tb_wid_ << ")" << std::endl;
        std::cout << "column per block " << column_per_block << std::endl;
        std::cout << "2d block dimension: (" << (nBody + column_per_block-1)/column_per_block << "," << (nBody + block.y-1)/block.y << ")" << std::endl;

        // calculate the initialia acceleration
        calculate_forces_2d<<<grid, block>>>(nBody, d_X[src_index], d_intermidiate_A, unroll_factor_, summation_result_per_body);
        simple_accumulate_intermidate_acceleration<<<nblocks, block_size_>>>(nBody, d_intermidiate_A, d_A[src_index], summation_result_per_body);
        timer.elapsed_previous("Calculated initial acceleration");

        {
            CORE::TIMER core_timer("all_iters");
            for (int i_iter = 0; i_iter < n_iter; i_iter++)
            {
                update_step_pos_f4<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_X[src_index], d_V[src_index], d_A[src_index], //input
                                                            d_X[dest_index], d_V_half);                                               // output

                cudaDeviceSynchronize();

                calculate_forces_2d<<<grid, block>>>(nBody, d_X[dest_index], d_intermidiate_A, unroll_factor_, summation_result_per_body);
                simple_accumulate_intermidate_acceleration<<<nblocks, block_size_>>>(nBody, d_intermidiate_A, d_A[dest_index], summation_result_per_body);

                cudaDeviceSynchronize();

                update_step_vel_f4<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_A[dest_index], d_V_half, //input
                                                            d_V[dest_index]);                                    // output
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

        std::ofstream X_file;
        std::ofstream V_file;
        std::ofstream A_file;
        X_file.open ("smallX.output");
        V_file.open ("smallV.output");
        A_file.open ("smallA.output");
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

        for(const auto i : {src_index, dest_index}){
            cudaFree(d_X[i]);
            cudaFree(d_V[i]);
            cudaFree(d_A[i]);
        }
        cudaFree(d_V_half);
        cudaDeviceReset();

        return system_state_result;
    }
}