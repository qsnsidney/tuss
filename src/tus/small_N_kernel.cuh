#pragma once
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

__global__ inline void update_step_vel_f4(unsigned nbody, data_t step_size, float4 *new_accer, data_t_3d *velocity_half,// new accer is accer at i+1 iteration
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
        data_t_3d x_self = make_data_t_3d(location[tid].x,location[tid].y,location[tid].z);
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

/*
 * The Functions below are taken from https://www.researchgate.net/publication/291770155_Fast_N-body_simulation_with_CUDA
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


__global__ inline void 
simple_accumulate_intermidate_acceleration(int N, float4* intermidiate_A, float4* output_A, int summation_res_per_body)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        float4 accumulated_accer = {0.0f, 0.0f, 0.0f, 0.0f}; 
        for (int i = 0; i < summation_res_per_body; i++) {
            accumulated_accer.x += intermidiate_A[tid * summation_res_per_body + i].x;
            accumulated_accer.y += intermidiate_A[tid * summation_res_per_body + i].y;
            accumulated_accer.z += intermidiate_A[tid * summation_res_per_body + i].z;
        }
        
        output_A[tid] = accumulated_accer;
    }
    
}

__global__ inline void
calculate_forces_2d(int N, float4 *globalX, float4 *globalA, int luf, int summation_res_per_body)
{
    extern __shared__ float4 shPosition[];
    float4 myPosition;

    int column_id = blockDim.x * blockIdx.x + threadIdx.x; // col
    int row_id = blockDim.y * blockIdx.y + threadIdx.y; // row

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
    for(int i = 0; i < num_element_shared_mem_read; i++) {
        // now, we need to be careful that shared_mem can't go overbound
        // in the caller, I pre allocate enough space in globalX (can some one help me to verify?)
        shPosition[shared_mem_offset + i] = globalX[global_offset + i];
    }

    // wait for all shared mem to be written
    __syncthreads();

    // don't forget that each thread is only reading a portion of the shared memory
    int shared_mem_read_offset = threadIdx.x * luf;

    // if the body is in the range. and the summation result is also in range
    // note that the block will end execution after the loop, so no syncthread is needed.
    if (row_id < N && column_id < summation_res_per_body)
    {
        for (int k = 0; k < luf; k++)
        {   
            //printf("shared mem location :%d, value: %f\n", shared_mem_read_offset + k, shPosition[shared_mem_read_offset + k]);
            acc = AccumulateBodyInteraction(myPosition, shPosition[shared_mem_read_offset + k], acc);
        }
        globalA[row_id * summation_res_per_body + column_id] = {acc.x, acc.y, acc.z, 0.0f};
    }
    // I decided to leave this code to profile how many threads are in idle along x dimension
    // if (row_id < N && column_id >= summation_res_per_body) {
    //     printf("%d, %d\n", row_id, column_id);
    // }
}

// Each thread reads 1 bank from the shared memory, but we limit its size (i.e. limit the # of rows)
// Data from this 1 bank can be shared between multiple bodies to perform accumulation in parallel
// We want 32 threads per block, since there are 32 banks in the shared memory
__global__ inline void
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
    
    if (gtid <= N-unrollFactor)
    {
        // we don't skip the object even if it's gtid > N.
        // reasons explained later.
        for (i = 0; i < unrollFactor; i++)
        {
            myPosition = globalX[gtid+i];
            acc[i] = {0.0f, 0.0f, 0.0f};

            // accumulate over 1 bank indicated by the theadIdx
            for (j = 0; j < N; j++) // j - shared mem row index
            {
                shPosition = globalX[32*j + threadIdx.x];
                // calculate accumulation
                acc[i] = AccumulateBodyInteraction(myPosition, shPosition, acc[i]);
            }
        }
        // Save the result in global memory for the integration step.
        __syncthreads();
        for (i = 0; i < unrollFactor; i++)
        {
            acc4 = globalA[gtid+i];
            globalA[gtid+i] = {acc[i].x+acc4.x, acc[i].y+acc4.y, acc[i].z+acc4.z, 0.0f};
        }
        __syncthreads();
    }  
}