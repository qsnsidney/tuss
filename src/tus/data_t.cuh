#pragma once

/*
 * Data type related helper function
 */
typedef float data_t;
typedef float3 data_t_3d;
// WARNING: this function has hardcoded assumption on float vs double
// inline function has to be defined in the header, otherwise, how does the compiler know
// what the inline function looks like before linker?
__host__ __device__ inline data_t_3d make_data_t_3d(const data_t a, const data_t b, const data_t c)
{
    return make_float3(a, b, c);
}

__host__ __device__ inline data_t_3d operator+(const data_t_3d &a, const data_t_3d &b)
{

    return make_data_t_3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline data_t_3d operator-(const data_t_3d &a, const data_t_3d &b)
{

    return make_data_t_3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline data_t_3d operator*(const data_t_3d &a, const data_t &b)
{

    return make_data_t_3d(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline data_t_3d operator/(const data_t_3d &a, const data_t &b)
{

    return make_data_t_3d(a.x / b, a.y / b, a.z / b);
}

// __host__ inline data_t gen_random_data_t(unsigned upper_bound)
// {
//     return (data_t)rand() / (data_t)(RAND_MAX / (2 * upper_bound)) - (data_t)upper_bound;
// }

// // randomly initialize the input array with type data_t in [-range, range)
// __host__ inline void random_initialize_vector(data_t_3d *input_array, size_t size, data_t range)
// {
//     for (size_t i = 0; i < size; i++)
//     {
//         input_array[i] = make_data_t_3d(gen_random_data_t(range), gen_random_data_t(range), gen_random_data_t(range));
//         //input_array[i] = make_data_t_3d(i, 2 * i, 3 * i);
//     }
// }

// __host__ inline void random_initialize_mass(data_t *input_array, size_t size, data_t range)
// {
//     //hack: + range at the end to offset the nagative
//     for (size_t i = 0; i < size; i++)
//     {
//         input_array[i] = gen_random_data_t(range) + range;
//         //input_array[i] = (float)(i) / 2;
//     }
// }

// WARNING: this function has hardcoded assumption on float vs double
// norm(a-b) ^ 1.5
__device__ inline data_t power_norm(data_t_3d a, data_t_3d b)
{
    data_t res = sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z) + CORE::UNIVERSE::epislon_square);
    return res * res * res;
}

// norm(a) ^ 1.5
__device__ inline data_t power_norm(data_t_3d a)
{
    data_t res = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + CORE::UNIVERSE::epislon_square);
    return res * res * res;
}

// WARNING: this function has hardcoded assumption on float vs double
__device__ inline data_t power_norm_inverse(data_t_3d a, data_t_3d b)
{
    data_t res = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z) + CORE::UNIVERSE::epislon_square;
    return 1.0f / sqrtf(res * res * res);
}