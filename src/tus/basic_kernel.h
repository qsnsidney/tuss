#pragma once
#include "Data_t.h"

__global__ void update_step(unsigned nbody, data_t step_size, data_t_3d *i_location, data_t_3d *i_velocity, data_t_3d *i_accer, data_t *mass, data_t_3d *new_accer, // new accer is accer at i+1 iteration
                            data_t_3d *o_location, data_t_3d *o_velocity);

__global__ void calculate_acceleration(unsigned nbody, data_t_3d *location, data_t *mass, data_t_3d *acceleration);