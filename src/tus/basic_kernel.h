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

__global__ inline void calculate_acceleration(unsigned nbody, data_t_3d *location, data_t *mass, data_t_3d *acceleration, int stride)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    data_t_3d points_to_calculate[2];
    data_t_3d acceleration_accums[2];
    
    // initialize
    for(unsigned offset = 0, i = 0; i < stride; ++i, offset += stride) {
        
        if(tid + offset > nbody) {
            continue;
        }
        
        points_to_calculate[i] = location[tid + offset];
        acceleration_accums[i] = make_data_t_3d(0, 0, 0);
    }
    
    // calculate force excerted by all other points
    for (unsigned j = 0; j < nbody; j++)
    {
        data_t_3d neighbour_location = location[j];
        data_t neighbour_mass = mass[j];
            
        for(unsigned offset = 0, i = 0; i < stride; ++i, offset += stride) {
            
            if(j == tid + offset) {
                continue;
            }
            
            data_t_3d numerator = (neighbour_location - points_to_calculate[i]) * neighbour_mass;
            data_t denominator = power_norm(points_to_calculate[i], neighbour_location);
            data_t_3d new_term = (numerator / denominator);
            acceleration_accums[i] = acceleration_accums[i] + new_term;
        }
           
        //printf("tid = %d, new_term %f, %f, %f\naccumulated_accer %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z, accumulated_accer.x, accumulated_accer.y, accumulated_accer.z);
    }
        
    for(unsigned offset = 0, i = 0; i < stride; ++i, offset += stride) {
        
        if(tid + offset > nbody) {
            continue;
        }
        
        acceleration[tid + offset] = acceleration_accums[i];
    }
}