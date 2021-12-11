#pragma once
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

__global__ inline void update_step_vel(unsigned nbody, data_t step_size, data_t *mass, data_t_3d *new_accer, data_t_3d *velocity_half,// new accer is accer at i+1 iteration
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
    float s = massj  / distSixth;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ inline float3
tile_calculation(float3 myPosition, float3 accel, int accum_length, float* devM, int offset)
{
    int i;
    extern __shared__ float3 shPosition[];
    for (i = 0; i < accum_length; i++) {
        // we don't need to check the object index.
        // because the vector subtration of oneself will just yields 0.
        // hence contributes no acceleration.
        accel = AccumulatebodyBodyInteraction(myPosition, shPosition[i], accel, devM[i + offset]);
    }
    return accel;
}

// each calculate forces handles one body
__global__ inline void
calculate_forces(int N, void *devX, float *devM, void *devA)
{
    extern __shared__ float3 shPosition[];
    float3 *globalX = (float3 *)devX;
    float3 *globalA = (float3 *)devA;
    float3 myPosition;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // we don't skip the object even if it's gtid > N.
    // reasons explained later.
    if(gtid < N){
        myPosition = globalX[gtid];
    } else {
        myPosition = {0.0f, 0.0f, 0.0f};
    }
    for (i = 0, tile = 0; i < N; i += blockDim.x, tile++) {
        
        // decide which piece of memory to read into the shared mem
        int idx = tile * blockDim.x + threadIdx.x;

        // printf("gid: %d. idx: %d\n",gtid, idx);
        // It is possible that the current mem to read is out of bound
        // but the thread itself is dealing with a valid body
        // for example, when there are 48 bodies with block_size = 32. 
        // in the 2nd iteration, thread of body 24 will try to read sharemem
        // of body 56. but we should not skip body 24's accleration accumulatio
        if(idx >= N) {
            shPosition[threadIdx.x] = {0.0f, 0.0f, 0.0f};
        }
        else {
            shPosition[threadIdx.x] = globalX[idx];
        }

        // we have to skip the thread that's greater than gtid here 
        // instead of earlier, because the thread could be reading some
        // shared mem data. imagine a case of block size = 8 and 9 body.
        // then the thread with gtid 9 will be reading the body1's location
        // in the first iteration. now the thread is done with loading the shared mem
        // so we can skip it.
        if(gtid >= N) {
            continue;
        }
        auto offset = tile * blockDim.x;
        // Ideally, we should take care of the case where the last tile contains less than 
        // num_block of data. only let the tiled function process min(blocksize, remaining elements) 
        // in length. but because we already load out of bound shared mem with 0s. we don't have to 
        // worry about out of bound anymore.
        __syncthreads();
        acc = tile_calculation(myPosition, acc, blockDim.x, devM, offset);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float3 acc3 = {acc.x, acc.y, acc.z};
    globalA[gtid] = acc3;
}