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

#include <cublas_v2.h>

__global__ void update_step_pos_f4(unsigned nbody, data_t step_size, float4 *i_location, data_t_3d *i_velocity, float3 *i_accer, // new accer is accer at i+1 iteration
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

__global__ void update_step_vel_f4(unsigned nbody, data_t step_size, float3 *new_accer, data_t_3d *velocity_half, // new accer is accer at i+1 iteration
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
 * The AccumulateBodyInteraction Function is taken from https://www.researchgate.net/publication/291770155_Fast_N-body_simulation_with_CUDA
 * with only changing sqrt to rsqrt
 */
__device__ inline float3
AccumulateBodyInteraction(float4 bi, float4 bj, float3 ai)
{
    // r_ij [3 FLOPS]
    float x_diff = bj.x - bi.x;
    float y_diff = bj.y - bi.y;
    float z_diff = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff + CORE::UNIVERSE::epislon_square;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = rsqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += x_diff * s;
    ai.y += y_diff * s;
    ai.z += z_diff * s;
    return ai;
}

__global__ void
simple_accumulate_intermidate_acceleration(int N, float4 *intermidiate_A, float4 *output_A, int summation_res_per_body)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N)
    {
        float3 accumulated_accer = make_float3(output_A[tid].x, output_A[tid].y, output_A[tid].z);
        for (int i = 0; i < summation_res_per_body; i++)
        {
            accumulated_accer.x += intermidiate_A[tid * summation_res_per_body + i].x;
            accumulated_accer.y += intermidiate_A[tid * summation_res_per_body + i].y;
            accumulated_accer.z += intermidiate_A[tid * summation_res_per_body + i].z;
        }

        output_A[tid] = make_float4(accumulated_accer.x, accumulated_accer.y, accumulated_accer.z, 0.0f);
    }
}

__global__ void
calculate_forces_2d(int N, size_t offset, float4 *globalX, void *outputA, int luf, int summation_res_per_body)
{
    extern __shared__ float4 shPosition[];
    float4 myPosition;
    int column_id = blockDim.x * blockIdx.x + threadIdx.x; // col
    int row_id = blockDim.y * blockIdx.y + threadIdx.y;    // row

    myPosition = globalX[row_id];
    float3 acc = {0.0f, 0.0f, 0.0f};

    // number of shared mem element populate to be done by each thread in a block.
    // for example. for a 64 * 4 block with luf = 1024.
    // each thread reads 1024 * 4 / (64 * 4) = 16 shared mem loc
    int num_element_shared_mem_read = luf / blockDim.y;

    // the beginning location of global offset to read memory from
    // column_id * luf accounts for the fact that each past column id already handles luf memory location
    // threadIdx.y * num_element_shared_mem_read is there because each luf is handled by
    // all thread on the same y dimension
    int global_offset = column_id * luf + threadIdx.y * num_element_shared_mem_read;

    // the offset of shared_mem to be populated by this exact thread in the block.
    // for example, in a 64 * 4 configuration. the (0,0) block handles the first 16 read
    // the (63,3) handles the last 16 reads. where (63, 3) => 4080
    int shared_mem_offset = threadIdx.x * luf + threadIdx.y * num_element_shared_mem_read;
    for (volatile int i = 0; i < num_element_shared_mem_read; i++)
    {
        // now, we need to be careful that shared_mem can't go overbound
        // in the caller, I pre allocate enough space in globalX (can some one help me to verify?)
        shPosition[shared_mem_offset + i] = globalX[offset + global_offset + i];
    }

    // wait for all shared mem to be written
    __syncthreads();

    // don't forget that each thread is only reading a portion of the shared memory
    int shared_mem_read_offset = threadIdx.x * luf;

    // if the body is in the range. and the summation result is also in range
    // note that the block will end execution after the loop, so no syncthread is needed.
    if (row_id < N && column_id < summation_res_per_body)
    {
        float* ptr = (float *)outputA;
        for (int k = 0; k < luf; k++)
        {
            //printf("shared mem location :%d, value: %f\n", shared_mem_read_offset + k, shPosition[shared_mem_read_offset + k]);
            acc = AccumulateBodyInteraction(myPosition, shPosition[shared_mem_read_offset + k], acc);
        }
        ptr[row_id * summation_res_per_body * 3 + column_id] = acc.x;
        ptr[row_id * summation_res_per_body * 3 + column_id + summation_res_per_body] = acc.y;
        ptr[row_id * summation_res_per_body * 3 + column_id + 2 * summation_res_per_body] = acc.z;
    }
    // I decided to leave this code to profile how many threads are in idle along x dimension
    // if (row_id < N && column_id >= summation_res_per_body) {
    //     printf("%d, %d\n", row_id, column_id);
    // }
}

__global__ void
calculate_forces_2d_no_conflict(int N, size_t offset, float4 *globalX, void *globalA, int luf, int summation_res_per_body)
{
    extern __shared__ float4 shPosition[];
    float4 myPosition;
    int column_id = blockDim.x * blockIdx.x + threadIdx.x; // col
    int row_id = blockDim.y * blockIdx.y + threadIdx.y;    // row

    myPosition = globalX[row_id];
    float3 acc = {0.0f, 0.0f, 0.0f};

    // number of shared mem element populate to be done by each thread in a block.
    // for example. for a 64 * 4 block with luf = 1024.
    // each thread reads 1024 * 4 / (64 * 4) = 16 shared mem loc
    int num_element_shared_mem_read = luf / blockDim.y;

    // the beginning location of global offset to read memory from
    // column_id * luf accounts for the fact that each past column id already handles luf memory location
    // threadIdx.y * num_element_shared_mem_read is there because each luf is handled by
    // all thread on the same y dimension
    int global_offset = column_id * luf + threadIdx.y * num_element_shared_mem_read;

    // some very hacky calculation
    const int warp_size = 32;
    int actual_thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = actual_thread_id % warp_size; // the id of thread in its warp belong to
    int num_wrap = blockDim.x * blockDim.y / warp_size;
    int thread_group = actual_thread_id / warp_size;                      // which warp does the thread belong to belong to
    int shared_mem_offset = thread_group * (luf * blockDim.x / num_wrap); // how many bytes are handled by 1 thread
    for (volatile int i = 0; i < num_element_shared_mem_read; i++)
    {
        // now, we need to be careful that shared_mem can't go overbound
        // in the caller, I pre allocate enough space in globalX (can some one help me to verify?)
        shPosition[shared_mem_offset + i * warp_size + warp_id] = globalX[offset + global_offset + i];
        // an version if bank conflict read is used in later loop
        //shPosition[shared_mem_offset + i * warp_size + warp_id] = globalX[offset + global_offset + shared_mem_offset + i * warp_size + warp_id];
    }

    // wait for all shared mem to be written
    __syncthreads();

    // if the body is in the range. and the summation result is also in range
    // note that the block will end execution after the loop, so no syncthread is needed.
    float* ptr = (float *)globalA;
    if (row_id < N && column_id < summation_res_per_body)
    {
        for (int k = 0; k < luf; k++)
        {
            // for a continuous memory region. do it as
            // | Pos1 | Pos2 | Pos3 | Pos4 |
            //   t1      t2     t3     t4
            acc = AccumulateBodyInteraction(myPosition, shPosition[k * blockDim.x + threadIdx.x], acc);
        }
        ptr[row_id * summation_res_per_body * 3 + column_id] = acc.x;
        ptr[row_id * summation_res_per_body * 3 + column_id + summation_res_per_body] = acc.y;
        ptr[row_id * summation_res_per_body * 3 + column_id + 2 * summation_res_per_body] = acc.z;
    }
}

// Each thread reads 1 bank from the shared memory, but we limit its size (i.e. limit the # of rows)
// Data from this 1 bank can be shared between multiple bodies to perform accumulation in parallel
// We want 32 threads per block, since there are 32 banks in the shared memory
__global__ void
calculate_forces_1d(int N, void *devX, void *devA, int p)
{
    //extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)devX;
    float4 *globalA = (float4 *)devA;
    float4 myPosition;
    float4 shPosition;
    int i, j;
    const int unrollFactor = 4;
    float3 acc[unrollFactor];
    float4 acc4;
    int gtid = unrollFactor * (blockIdx.x * blockDim.x + threadIdx.x);

    if (gtid <= N - unrollFactor)
    {
        // we don't skip the object even if it's gtid > N.
        // reasons explained later.
        for (i = 0; i < unrollFactor; i++)
        {
            myPosition = globalX[gtid + i];
            acc[i] = {0.0f, 0.0f, 0.0f};

            // accumulate over 1 bank indicated by the theadIdx
            for (j = 0; j < N; j++) // j - shared mem row index
            {
                shPosition = globalX[32 * j + threadIdx.x];
                // calculate accumulation
                acc[i] = AccumulateBodyInteraction(myPosition, shPosition, acc[i]);
            }
        }
        // Save the result in global memory for the integration step.
        __syncthreads();
        for (i = 0; i < unrollFactor; i++)
        {
            acc4 = globalA[gtid + i];
            globalA[gtid + i] = {acc[i].x + acc4.x, acc[i].y + acc4.y, acc[i].z + acc4.z, 0.0f};
        }
        __syncthreads();
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

        // number of body for the problem
        size_t nBody = system_state_snapshot().size();

        // number of body to accumulate field for each kernel call.
        // each kernel call accumluate AccumBody's acceleration for all Nbodies.
        size_t AccumBody = nBody;
        if (AccumBody > 100000)
        {
            AccumBody = 100000;
        }
        /* BIN file of initial conditions */
        const auto &ic = system_state_snapshot();

        dim3 block(tb_len_, tb_wid_);
        int column_per_block = (tb_len_ * unroll_factor_);
        assert(unroll_factor_ % tb_wid_ == 0);
        int num_block_x_dim = (AccumBody + column_per_block - 1) / column_per_block;
        int num_block_y_dim = (nBody + block.y - 1) / block.y;
        dim3 grid(num_block_x_dim, num_block_y_dim);

        std::cout << "2d block dimension: (" << block.x << "," << block.y << ")" << std::endl;
        std::cout << "column per block " << column_per_block << std::endl;
        std::cout << "2d grid dimension: (" << grid.x << "," << grid.y << ")" << std::endl;

        // size
        size_t vector_size_3d = sizeof(data_t_3d) * nBody;
        size_t vector_size_4d = sizeof(float4) * nBody;

        // to make boundary check not so painful, pre allocated extra memory so each thread doesn't need to worry about
        // boundary checking
        size_t position_quantized_element = (nBody + column_per_block - 1) / column_per_block * column_per_block;
        size_t quantized_accum_body = (nBody + (AccumBody - 1)) / AccumBody * AccumBody;
        size_t gter = get_max(position_quantized_element, quantized_accum_body);
        size_t vector_size_4d_qtzed = sizeof(float4) * gter;
        size_t num_loop = quantized_accum_body / AccumBody;
        std::cout << "quantize to " << gter << std::endl;
        printf("summation expects to take %d / %d = %d iteration\n", quantized_accum_body, AccumBody, num_loop);
        /*
         *   host side memory allocation
         */
        data_t_3d *h_V, *h_output_V;
        float4 *h_X, *h_output_X;
        float3 *h_A;

        host_malloc_helper((void **)&h_V, vector_size_3d);
        host_malloc_helper((void **)&h_output_V, vector_size_3d);

        host_malloc_helper((void **)&h_X, vector_size_4d_qtzed);
        host_malloc_helper((void **)&h_A, vector_size_3d);
        host_malloc_helper((void **)&h_output_X, vector_size_4d);

        timer.elapsed_previous("allocated host side memory");
        /*
         *   input randome initialize
         */
        for (int i = 0; i < num_block_x_dim * column_per_block; i++)
        {
            h_X[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        parse_ic_f4(h_X, h_V, ic);
        timer.elapsed_previous("deserialize_body_state_vec_from_csv");

        /*
         * create double buffer on device side
         */
        float4 **d_X;
        float3 **d_A;
        unsigned src_index = 0;
        unsigned dest_index = 1;

        // for each kernel call, how many intermidate sum should we cache in global memory
        int summation_result_per_body = (AccumBody + unroll_factor_ - 1) / unroll_factor_;
        std::cout << "summation result per body is " << summation_result_per_body << std::endl;
        float4 *d_intermidiate_A;
        gpuErrchk(cudaMalloc((void **)&d_intermidiate_A, sizeof(float3) * nBody * summation_result_per_body));

        d_X = (float4 **)malloc(2 * sizeof(float4 *));
        gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size_4d_qtzed));
        gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size_4d_qtzed));

        d_A = (float3 **)malloc(2 * sizeof(float3 *));
        gpuErrchk(cudaMalloc((void **)&d_A[src_index], vector_size_3d));
        gpuErrchk(cudaMalloc((void **)&d_A[dest_index], vector_size_3d));

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
        cudaMemcpy(d_X[src_index], h_X, vector_size_4d_qtzed, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V[src_index], h_V, vector_size_3d, cudaMemcpyHostToDevice);
        timer.elapsed_previous("copied input data from host to device");

        // nthread is assigned to either 32 by default or set to a custom power of 2 by user
        std::cout << "Set thread_per_block to " << block_size_ << std::endl;
        unsigned nblocks = (nBody + block_size_ - 1) / block_size_;

        std::cout << "set number of body to " << nBody << std::endl;
        std::cout << "using " << column_per_block * sizeof(float4) << " bytes per block" << std::endl;
        // I would highly recommend be careful about setting shared mem > 16384
        assert(column_per_block * sizeof(float4) <= 16384);
        // calculate the initialia acceleration
        cudaMemset(d_A[src_index], 0, vector_size_3d);

        /***************/
        /* APPROACH #4 */
        /***************/
        cublasHandle_t handle;
        cublasCreate(&handle);
        int Nrows = nBody * 3;
        int Ncols = summation_result_per_body;
        float alpha = 1.f;
        float beta  = 1.f;
        float *d_column_one_matrix, *h_column_one_matrix;
        host_malloc_helper((void **)&h_column_one_matrix, sizeof(float) * Ncols);
        for(int i = 0; i < Ncols; i++) {
            h_column_one_matrix[i] = 1.0f;
        }
        gpuErrchk(cudaMalloc((void **)&d_column_one_matrix, sizeof(float) * Ncols));
        cudaMemcpy(d_column_one_matrix, h_column_one_matrix, sizeof(float) * Ncols, cudaMemcpyHostToDevice);

        for (int i = 0; i < num_loop; i++)
        {
            size_t offset = i * AccumBody;
            if (block.x == 1)
            {
                calculate_forces_2d<<<grid, block, column_per_block * sizeof(float4)>>>(nBody, offset, d_X[src_index], d_intermidiate_A, unroll_factor_, summation_result_per_body);
            }
            else
            {
                calculate_forces_2d_no_conflict<<<grid, block, column_per_block * sizeof(float4)>>>(nBody, offset, d_X[src_index], d_intermidiate_A, unroll_factor_, summation_result_per_body);
            }

            // int num_column = summation_result_per_body;
            // //                                                            y dim     x dim
            // float * h_intermidiate_A = (float*)malloc(sizeof(float4) * nBody * num_column);
            // cudaMemcpy(h_intermidiate_A, d_intermidiate_A, sizeof(float4) * nBody * num_column, cudaMemcpyDeviceToHost);
            // for(int i = 0; i < nBody * 4; i++) {
            //     float accum = 0;
            //     for (int j = 0; j < num_column; j++) {
            //         accum += h_intermidiate_A[j + i * num_column];
            //         printf("%f ", h_intermidiate_A[j + i * num_column]);
            //     }
            //     printf("accum = %f\n", accum);
            // }

            cublasSgemv(handle, CUBLAS_OP_T, Ncols, Nrows, &alpha, (float *)d_intermidiate_A, Ncols, 
                               (float *)d_column_one_matrix, 1, &beta, (float *) d_A[src_index], 1);
            //cublasSgemv(handle, CUBLAS_OP_N, Nrows, Ncols, &alpha, (float *)d_intermidiate_A, Nrows, 
                               //(float *)d_column_one_matrix, 1, &beta, (float *) d_A[src_index], 1);
        }
        timer.elapsed_previous("Calculated initial acceleration");

        {
            CORE::TIMER core_timer("all_iters");
            for (int i_iter = 0; i_iter < n_iter; i_iter++)
            {
                update_step_pos_f4<<<nblocks, block_size_>>>(nBody, (data_t)dt(), d_X[src_index], d_V[src_index], d_A[src_index], //input
                                                               d_X[dest_index], d_V_half);                                          // output

                cudaDeviceSynchronize();
                cudaMemset(d_A[dest_index], 0, vector_size_3d);
                for (int i = 0; i < num_loop; i++)
                {
                    size_t offset = i * AccumBody;
                    if (block.x == 1)
                    {
                        calculate_forces_2d<<<grid, block, column_per_block * sizeof(float4)>>>(nBody, offset, d_X[dest_index], d_intermidiate_A, unroll_factor_, summation_result_per_body);
                    }
                    else
                    {
                        calculate_forces_2d_no_conflict<<<grid, block, column_per_block * sizeof(float4)>>>(nBody, offset, d_X[dest_index], d_intermidiate_A, unroll_factor_, summation_result_per_body);
                    }
                    cublasSgemv(handle, CUBLAS_OP_T, Ncols, Nrows, &alpha, (float *)d_intermidiate_A, Ncols, 
                               (float *)d_column_one_matrix, 1, &beta, (float *) d_A[dest_index], 1);
                }
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
        cudaMemcpy(h_A, d_A[src_index], vector_size_3d, cudaMemcpyDeviceToHost);

        std::ofstream X_file;
        std::ofstream V_file;
        std::ofstream A_file;
        X_file.open("smallX.output");
        V_file.open("smallV.output");
        A_file.open("smallA.output");
        for (int i = 0; i < nBody; i++)
        {
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