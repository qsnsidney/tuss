#pragma once
#include "constant.h"
/*
 * Data type related helper function
 */
typedef float data_t;
typedef float3 data_t_3d;
// WARNING: this function has hardcoded assumption on float vs double
__host__ __device__ data_t_3d make_data_t_3d(const data_t a, const data_t b, const data_t c);

__host__ __device__ data_t_3d operator+(const data_t_3d &a, const data_t_3d &b);

__host__ __device__ data_t_3d operator-(const data_t_3d &a, const data_t_3d &b);

__host__ __device__ data_t_3d operator*(const data_t_3d &a, const data_t &b);

__host__ __device__ data_t_3d operator/(const data_t_3d &a, const data_t &b);

__host__ data_t gen_random_data_t(unsigned upper_bound);

// randomly initialize the input array with type data_t in [-range, range)
__host__ void random_initialize_vector(data_t_3d *input_array, size_t size, data_t range);

__host__ void random_initialize_mass(data_t *input_array, size_t size, data_t range);

// WARNING: this function has hardcoded assumption on float vs double
__device__ data_t power_norm(data_t_3d a, data_t_3d b);