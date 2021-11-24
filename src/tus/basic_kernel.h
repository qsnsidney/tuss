#pragma once
#include "data_t.h"

__global__ inline void update_step(unsigned nbody, data_t step_size, data_t_3d *i_location, data_t_3d *i_velocity, data_t_3d *i_accer, data_t *mass, data_t_3d *new_accer, // new accer is accer at i+1 iteration
                            data_t_3d *o_location, data_t_3d *o_velocity)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        // v1/2          =         vi      +     ai  *          1/2 *    dt
        data_t_3d v_half = i_velocity[tid] + i_accer[tid] * ((data_t)0.5 * step_size);
        // Xi+1         =      xi         +       vi        *     dt    +    ai   *     1/2     *     (dt)^2
        o_location[tid] = i_location[tid] + i_velocity[tid] * step_size + i_accer[tid] * (data_t)0.5 * powf(step_size, 2);
        // Vi+1         =  V1/2  +      ai+1      *     1/2      *    dt
        o_velocity[tid] = v_half + new_accer[tid] * ((data_t)0.5 * step_size);
    }
}

__global__ inline void calculate_acceleration(unsigned nbody, data_t_3d *location, data_t *mass, data_t_3d *acceleration, int n)
{
    extern __shared__ data_t_3d shared_state[];
    data_t_3d *shared_locations = (data_t_3d*) shared_state;
    data_t *shared_mass = (data_t*) &shared_locations[n];
    
    unsigned block_tid_start = blockDim.x * blockIdx.x;
    unsigned block_tid_end = block_tid_start + blockIdx.x;
    
    unsigned tid = block_tid_start + threadIdx.x;
    if (tid < nbody)
    {
        shared_locations[threadIdx.x] = location[tid];
        shared_mass[threadIdx.x] = mass[tid];
        
        data_t_3d accumulated_accer = make_data_t_3d(0, 0, 0);
        data_t_3d x_self = shared_locations[threadIdx.x];
        for (unsigned j = 0; j < nbody; j++)
        {

            //Offset neighbour fetching by thread id
            int index = (tid + j) % nbody;
            
            if (index == tid)
            {
                continue;
            }
            
            //Check if current neighbour belonging to local block
            //If true, read from block shared memory
            if( index >= block_tid_start and index < block_tid_end ) {
                int local_index = index - block_tid_start;
                data_t_3d numerator = (x_self - shared_locations[local_index]) * shared_mass[local_index];
                data_t denominator = power_norm(x_self, shared_locations[local_index]);
                data_t_3d new_term = (numerator / denominator);
                accumulated_accer = accumulated_accer + new_term;
            //Else read from global memory
            } else {    
                data_t_3d numerator = (x_self - location[index]) * mass[index];
                data_t denominator = power_norm(x_self, location[index]);
                data_t_3d new_term = (numerator / denominator);       
                accumulated_accer = accumulated_accer + new_term;         
            }  
            //printf("tid = %d, new_term %f, %f, %f\naccumulated_accer %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z, accumulated_accer.x, accumulated_accer.y, accumulated_accer.z);
        }
        acceleration[tid] = accumulated_accer;
    }
}