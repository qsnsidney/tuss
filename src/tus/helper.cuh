#pragma once
#include "core/macros.hpp"
#include "data_t.cuh"
#include "core/physics.hpp"
#include "core/serde.h"
#include <iostream>
#include <fstream>

inline size_t get_max(size_t a, size_t b)
{
    if(a > b) {
        return a;
    }
    return b;
}

inline void swap(unsigned &a, unsigned &b)
{
    unsigned temp = a;
    a = b;
    b = temp;
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

    std::cout << "Error: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if (abort)
        exit(code);
}

// helper function to allocate cuda host memory
inline void host_malloc_helper(void **ptr, size_t size)
{
    cudaError_t err = cudaMallocHost((void **)ptr, size);
    if (cudaSuccess != err)
    {
        std::cout << "Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(1);
    }
}

// Leverage from https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
inline __host__ bool IsPowerOfTwo(unsigned x)
{
    return (x & (x - 1)) == 0;
}

inline __host__ void parse_ic(data_t_3d *input_x, data_t_3d *input_v, data_t *input_m, const CORE::SYSTEM_STATE &ic)
{
    size_t length_to_parse = ic.size();
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

inline __host__ void parse_ic_f4(float4 *input_x, data_t_3d *input_v, const CORE::SYSTEM_STATE &ic)
{
    size_t length_to_parse = ic.size();
    std::cout << "parsing " << length_to_parse << " bodies\n";
    for (size_t i = 0; i < length_to_parse; i++)
    {
        CORE::POS p = std::get<CORE::POS>(ic[i]);
        CORE::VEL v = std::get<CORE::VEL>(ic[i]);
        CORE::MASS m = std::get<CORE::MASS>(ic[i]);
        input_x[i] = make_float4(p.x, p.y, p.z, m);
        input_v[i] = make_data_t_3d((data_t)v.x, (data_t)v.y, (data_t)v.z);
    }
}

inline __host__ void write_debug_output(std::string engine_name, float4 * X_output, data_t_3d * V_output, float4 * A_output, size_t nBody) {
#if 1
    std::ofstream X_file;
    std::ofstream V_file;
    std::ofstream A_file;
    X_file.open(engine_name + "_X.output");
    V_file.open(engine_name + "_V.output");
    A_file.open(engine_name + "_A.output");
    for(int i = 0; i < nBody; i++) {
        X_file << X_output[i].x << "\n";
        X_file << X_output[i].y << "\n";
        X_file << X_output[i].z << "\n";
        V_file << V_output[i].x << "\n";
        V_file << V_output[i].y << "\n";
        V_file << V_output[i].z << "\n";
        A_file << A_output[i].x << "\n";
        A_file << A_output[i].y << "\n";
        A_file << A_output[i].z << "\n";
    }
    X_file.close();
    V_file.close();
    A_file.close();
#endif
}