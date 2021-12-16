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
 * Functions below are taken from https://www.researchgate.net/publication/291770155_Fast_N-body_simulation_with_CUDA with
 * only necessary modifications such as boundary condition check and parameter fixing.
 * The code is purely intended to be used as a reference for performance comparsion.
 *
 */

__device__ inline float3
AccumulatebodyBodyInteraction(float4 bi, float4 bj, float3 ai)
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
    float invDistCube = 1.0f/sqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

// a version that exact matches with simple test bench
__device__ inline float3
AccumulatebodyBodyInteraction_exact_match(float4 bi, float4 bj, float3 ai)
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
    float s = bj.w  / distSixth;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ inline float3
tile_calculation(float4 myPosition, float3 accel, int accum_length)
{
    int i;
    extern __shared__ float4 shPosition[];
    for (i = 0; i < accum_length; i++) {
        // we don't need to check the object index.
        // because the vector subtration of oneself will just yields 0.
        // hence contributes no acceleration.
        accel = AccumulatebodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}

// Each thread reads 1 bank from the shared memory, but we limit its size (i.e. limit the # of rows)
// Data from this 1 bank can be shared between multiple bodies to perform accumulation in parallel
// We want 32 threads per block, since there are 32 banks in the shared memory
__global__ inline void
calculate_forces(int N, void *devX, void *devA, int p)
{
    //extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)devX;
    float4 *globalA = (float4 *)devA;
    float4 myPosition;
    float4 shPosition;
    int i, j;
    float3 acc[];
    float4 acc4;
    int unrollFactor = 4;
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
                acc[i] = AccumulatebodyBodyInteraction(myPosition, shPosition, acc[i]);
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

// each calculate forces handles one body
__global__ inline void
calculate_forces_tiling(int N, void *devX, void *devA, int p)
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
    for (i = 0, tile = 0; i < N; i += blockDim.x, tile++) {
        
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
        acc = tile_calculation(myPosition, acc, blockDim.x);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
}