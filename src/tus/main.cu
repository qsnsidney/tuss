#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <iostream>

#include "physics.h"
#include "csv.h"

// Comment out this line to enable debug mode
// #define NDEBUG

#define RANDOM_RANGE 5
#define EPSILON 0.01
#define SIM_TIME 10
#define STEP_SIZE 1

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

/*
 * Data type related helper function
 */
typedef float data_t;
typedef float3 data_t_3d;
// WARNING: this function has hardcoded assumption on float vs double
__host__ __device__ inline data_t_3d make_data_t_3d(const data_t a, const data_t b, const data_t c)
{
    return make_float3(a, b, c);
}

__host__ __device__ data_t_3d operator+(const data_t_3d &a, const data_t_3d &b)
{

    return make_data_t_3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ data_t_3d operator-(const data_t_3d &a, const data_t_3d &b)
{

    return make_data_t_3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ data_t_3d operator*(const data_t_3d &a, const data_t &b)
{

    return make_data_t_3d(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ data_t_3d operator/(const data_t_3d &a, const data_t &b)
{

    return make_data_t_3d(a.x / b, a.y / b, a.z / b);
}

__host__ data_t gen_random_data_t(unsigned upper_bound)
{
    return (data_t)rand() / (data_t)(RAND_MAX / (2 * upper_bound)) - (data_t)upper_bound;
}

// randomly initialize the input array with type data_t in [-range, range)
__host__ void random_initialize_vector(data_t_3d *input_array, size_t size, data_t range)
{
    for (size_t i = 0; i < size; i++)
    {
        input_array[i] = make_data_t_3d(gen_random_data_t(range), gen_random_data_t(range), gen_random_data_t(range));
        //input_array[i] = make_data_t_3d(i, 2 * i, 3 * i);
    }
}

__host__ void random_initialize_mass(data_t *input_array, size_t size, data_t range)
{
    //hack: + range at the end to offset the nagative
    for (size_t i = 0; i < size; i++)
    {
        input_array[i] = gen_random_data_t(range) + range;
        //input_array[i] = (float)(i) / 2;
    }
}

__host__ void parse_ic(data_t_3d *input_v, data_t_3d *input_x, std::vector<CORE::POS_VEL_PAIR>& ic)
{
    for (size_t i = 0; i < ic.size(); i++)
    {   
        CORE::POS p = ic[i].first;
        CORE::VEL v = ic[i].second;
        
        input_x[i] = make_data_t_3d((data_t)p.x, (data_t)p.y, (data_t)p.z);
        input_v[i] = make_data_t_3d((data_t)v.x, (data_t)v.y, (data_t)v.z);
    }
}

// WARNING: this function has hardcoded assumption on float vs double
__device__ data_t power_norm(data_t_3d a, data_t_3d b)
{
    return powf(powf(a.x - b.x, 2) + powf(a.y - b.y, 2) + powf(a.z - b.z, 2) + powf(EPSILON, 2), 1.5);
}

/*
 * Here starts the actual kernel implemetation
 */

// temporary throw a dummy kernel just for very basic level sanity check
__global__ void kernel_place_holder(data_t_3d *input_ptr, data_t_3d *input_ptr2, data_t_3d *output_ptr, unsigned nsize)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nsize)
    {
        output_ptr[tid] = input_ptr2[tid] + input_ptr[tid];
    }
}

__global__ void update_step(unsigned nbody, data_t step_size, data_t_3d *i_location, data_t_3d *i_velocity, data_t_3d *i_accer, data_t *mass, data_t_3d *new_accer, // new accer is accer at i+1 iteration
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

__global__ void calculate_acceleration(unsigned nbody, data_t_3d *location, data_t *mass, data_t_3d *acceleration)
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
            data_t_3d numerator = (x_self - location[j]) * mass[j];
            data_t denominator = power_norm(x_self, location[j]);
            data_t_3d new_term = (numerator / denominator);
            accumulated_accer = accumulated_accer + new_term;
            //printf("tid = %d, new_term %f, %f, %f\naccumulated_accer %f, %f, %f\n", tid, new_term.x, new_term.y, new_term.z, accumulated_accer.x, accumulated_accer.y, accumulated_accer.z);
        }
        acceleration[tid] = accumulated_accer;
    }
}

int main(int argc, char *argv[])
{
    CORE::VEL v{1.0f, 2.0f, 3.0f};
    v *= 2.0f;
    std::cout << v << std::endl;

    /* Get Dimension */
    /// TODO: Add more arguments for input and output
    /// Haiqi: I think it should be "main [num_body] [simulation_end_time] [num_iteration] or [step_size]". or we simply let step_size = 1
    if (argc != 3)
    {
        printf("Error: The number of arguments is not exactly 2\n");
        return 0;
    }
    unsigned nBody = atoi(argv[1]);
    // temporarily assign them to MARCO
    unsigned simulation_time = SIM_TIME;
    unsigned step_size = STEP_SIZE;
    
    /* CSV files of initial conditions */
    std::string csv_path(argv[2]);
    
    srand(time(NULL));
    size_t vector_size = sizeof(data_t_3d) * nBody;
    size_t data_size = sizeof(data_t) * nBody;

    /*
     *   host side memory allocation
     */
    data_t_3d *h_X, *h_A, *h_V, *h_output_X;
    data_t *h_M;
    host_malloc_helper((void **)&h_X, vector_size);
    host_malloc_helper((void **)&h_A, vector_size);
    host_malloc_helper((void **)&h_V, vector_size);
    host_malloc_helper((void **)&h_output_X, vector_size);
    host_malloc_helper((void **)&h_M, data_size);
    
    /*
     *   input randome initialize
     */
    
    auto ic = CORE::parse_body_ic_from_csv(csv_path);   
    parse_ic(h_V, h_X, ic);
    
    /*
     *   input randome initialize
     */
    random_initialize_mass(h_M, nBody, RANDOM_RANGE);

    /*
     *  mass 
     */
    data_t *d_M;
    gpuErrchk(cudaMalloc((void **)&d_M, data_size));
    /*
     *   create double buffer on device side
     */
    data_t_3d **d_X, **d_A, **d_V;
    unsigned src_index = 0;
    unsigned dest_index = 1;
    d_X = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
    gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size));
    gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size));

    d_A = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
    gpuErrchk(cudaMalloc((void **)&d_A[src_index], vector_size));
    gpuErrchk(cudaMalloc((void **)&d_A[dest_index], vector_size));

    d_V = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
    gpuErrchk(cudaMalloc((void **)&d_V[src_index], vector_size));
    gpuErrchk(cudaMalloc((void **)&d_V[dest_index], vector_size));

    /*
     *   create double buffer on device side
     */
    // cudaMemcpy(d_A[0], h_A, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X[src_index], h_X, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V[src_index], h_V, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, data_size, cudaMemcpyHostToDevice);

    unsigned nthreads = 256;
    unsigned nblocks = (nBody + nthreads - 1) / nthreads;

    // calculate the initialia acceleration
    calculate_acceleration<<<nblocks, nthreads>>>(nBody, d_X[src_index], d_M, d_A[src_index]);

    for (unsigned step = 0; step < simulation_time; step += step_size)
    {
        // There should be more than one ways to do synchronization. I temporarily randomly choosed one
        calculate_acceleration<<<nblocks, nthreads>>>(nBody, d_X[src_index], d_M,                                                          //input
                                                      d_A[dest_index]);                                                                    // output
        update_step<<<nblocks, nthreads>>>(nBody, (data_t)step_size, d_X[src_index], d_V[src_index], d_A[src_index], d_M, d_A[dest_index], //input
                                           d_X[dest_index], d_V[dest_index]);                                                              // output

        swap(src_index, dest_index);
    }

    // at end, the final data is actually at src_index because the last swap
    cudaMemcpy(h_output_X, d_X[src_index], vector_size, cudaMemcpyDeviceToHost);

    // Just for debug purpose on small inputs
    for (unsigned i = 0; i < nBody; i++)
    {
        printf("object = %d, %f, %f, %f\n", i, h_output_X[i].x, h_output_X[i].y, h_output_X[i].z);
    }

    //for(int i = 0; i < nBody; i++){
    //    printf("locations: tid = %d, %f, %f, %f\n", i, h_X[i].x, h_X[i].y, h_X[i].z);
    //    printf("mass: tid = %d, %f\n", i, h_M[i]);
    //}

    return 0;
}