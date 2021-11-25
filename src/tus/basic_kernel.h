#pragma once
#include "data_t.h"

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
    }
}

__global__ inline void update_step_vel(unsigned nbody, data_t step_size, data_t *mass, data_t_3d *new_accer, data_t_3d *velocity_half,// new accer is accer at i+1 iteration
                            data_t_3d *o_velocity)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nbody)
    {
        o_velocity[tid] = velocity_half[tid] + new_accer[tid] * ((data_t)0.5 * step_size);
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
            //printf("tid = %d, new_term %f, %f, %f\naccumulated_accer %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z, accumulated_accer.x, accumulated_accer.y, accumulated_accer.z);
        }
        acceleration[tid] = accumulated_accer;
    }
}