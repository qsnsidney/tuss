#pragma once
#include <sys/time.h>
#include "data_t.h"
#include "core/physics.hpp"
#include "core/serde.h"
#include <iostream>
void swap(unsigned &a, unsigned &b)
{
    unsigned temp = a;
    a = b;
    b = temp;
}

/* time stamp function in milliseconds */
__host__ double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

/*You can use the following for any CUDA function that returns cudaError_t type*/
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code == cudaSuccess)
        return;

    fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
        exit(code);
}

// helper function to allocate cuda host memory
void host_malloc_helper(void **ptr, size_t size)
{
    cudaError_t err = cudaMallocHost((void **)ptr, size);
    if (cudaSuccess != err)
    {
        printf("Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
}

// Leverage from https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
__host__ bool IsPowerOfTwo(unsigned x)
{
    return (x & (x - 1)) == 0;
}

__host__ void parse_ic(data_t_3d *input_x, data_t_3d *input_v, data_t *input_m, CORE::BODY_STATE_VEC &ic, size_t parse_length = 0)
{
    size_t length_to_parse = ic.size();
    if (parse_length != 0)
    {
        assert(parse_length <= length_to_parse);
        length_to_parse = parse_length;
    }
    std::cout << "parsing " << length_to_parse << " bodies\n";
    for (size_t i = 0; i < length_to_parse; i++)
    {
        CORE::POS p = std::get<CORE::POS>(ic[i]);
        CORE::VEL v = std::get<CORE::VEL>(ic[i]);
        CORE::MASS m = std::get<CORE::MASS>(ic[i]);
        input_x[i] = make_data_t_3d((data_t)p.x, (data_t)p.y, (data_t)p.z);
        input_v[i] = make_data_t_3d((data_t)v.x, (data_t)v.y, (data_t)v.z);
        input_m[i] = (data_t)m;
    }
}