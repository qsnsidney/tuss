#pragma once
#include <sys/time.h>
#include "data_t.h"
#include "physics.h"
#include "serde.h"

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

__host__ void parse_ic(data_t_3d *input_v, data_t_3d *input_x, const std::vector<CORE::BODY_IC> &ic)
{
    int i = 0;
    for (const auto [p, v, _mass] & : ic)
    {
        input_x[i] = make_data_t_3d((data_t)p.x, (data_t)p.y, (data_t)p.z);
        input_v[i] = make_data_t_3d((data_t)v.x, (data_t)v.y, (data_t)v.z);
        i++;
    }
}