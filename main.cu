#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Comment out this line to enable debug mode
// #define NDEBUG

/* time stamp function in milliseconds */
__host__ double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

/*You can use the following for any CUDA function that returns cudaError_t type*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code == cudaSuccess) return;
    
    fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

// helper function to allocate cuda host memory
void host_malloc_helper(void** ptr, size_t size){
    cudaError_t err = cudaMallocHost((void**)ptr, size);
    if(cudaSuccess != err) {
        printf("Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(1);
    }
}

/*
 * Data type related helper function
 */
typedef float data_t;
typedef float3 data_t_3d;  
__host__ __device__ inline data_t_3d make_data_t_3d(const data_t a, const data_t b, const data_t c) {
    return make_float3(a, b, c);
}

__host__ __device__ data_t_3d operator+(const data_t_3d & a, const data_t_3d & b) {
  
  return make_data_t_3d(a.x+b.x, a.y+b.y, a.z+b.z);

}

#define RANDOM_RANGE 5

__host__ data_t gen_random_data_t(unsigned upper_bound) {
    return (data_t)rand()/(data_t)(RAND_MAX/(2 * upper_bound)) - (data_t)upper_bound;
}

// randomly initialize the input array with type data_t in [-range, range)
__host__ void random_initialize_vector(data_t_3d * input_array, size_t size, data_t range) {
    for(size_t i = 0; i < size; i++){
        input_array[i] = make_data_t_3d(gen_random_data_t(range), gen_random_data_t(range), gen_random_data_t(range));
    }        
}

__host__ void random_initialize_mass(data_t * input_array, size_t size, data_t range){
    //hack: + range at the end to offset the nagative
    for(size_t i = 0; i < size; i++){
        input_array[i] = gen_random_data_t(range) + range;
    }
}

/*
 * Here starts the actual kernel implemetation
 */

// temporary throw a dummy kernel just for very basic level sanity check
__global__ void kernel_place_holder(data_t_3d * input_ptr, data_t_3d * input_ptr2, data_t_3d * output_ptr, int nsize) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nsize){
        output_ptr[tid] = input_ptr2[tid] + input_ptr[tid];
        //printf("filling in %d, %f, %f, %f\n", tid, input_ptr2[tid].x, input_ptr[tid].x, output_ptr[tid].x);
    }
}

int main(int argc, char *argv[])
{
    /* Get Dimension */
    /// TODO: Add more arguments for input and output
    if (argc != 2)
    {
        printf("Error: The number of arguments is not exactly 1\n");
        return 0;
    }
    unsigned nBody = atoi(argv[1]);
    srand(time(NULL)); 
    size_t vector_size = sizeof(data_t_3d) * nBody;
    size_t data_size = sizeof(data_t) * nBody;
    
    /*
     *   host side memory allocation
     */ 
    data_t_3d * h_X, * h_A, * h_V, *h_output_X;
    data_t * h_M;
    host_malloc_helper((void**)&h_X, vector_size);
    host_malloc_helper((void**)&h_A, vector_size);
    host_malloc_helper((void**)&h_V, vector_size);
    host_malloc_helper((void**)&h_output_X, vector_size);
    host_malloc_helper((void**)&h_M, data_size);
    
    /*
     *   input randome initialize
     */ 
    random_initialize_vector(h_X, nBody, RANDOM_RANGE);
    random_initialize_vector(h_V, nBody, RANDOM_RANGE);
    random_initialize_mass(h_M, nBody, RANDOM_RANGE);
    
    /*
     *   create double buffer on device side
     */ 
    data_t_3d ** d_X, ** d_A, ** d_V;
    d_X = (data_t_3d**)malloc(2 * sizeof(data_t_3d*));
    gpuErrchk(cudaMalloc((void**)&d_X[0], vector_size));
    gpuErrchk(cudaMalloc((void**)&d_X[1], vector_size));

    d_A = (data_t_3d**)malloc(2 * sizeof(data_t_3d*));
    gpuErrchk(cudaMalloc((void**)&d_A[0], vector_size));
    gpuErrchk(cudaMalloc((void**)&d_A[1], vector_size));    
    
    d_V = (data_t_3d**)malloc(2 * sizeof(data_t_3d*));
    gpuErrchk(cudaMalloc((void**)&d_V[0], vector_size));
    gpuErrchk(cudaMalloc((void**)&d_V[1], vector_size));

    /*
     *   create double buffer on device side
     */ 
    // cudaMemcpy(d_A[0], h_A, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X[0], h_X, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V[0], h_V, vector_size, cudaMemcpyHostToDevice);

    int nblocks = 4, nthreads=4;
    kernel_place_holder<<<nblocks, nthreads>>>(d_X[0], d_A[0], d_V[0], nBody);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output_X, d_V[0], vector_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < nBody; i++){
        //printf("tid = %d, %f, %f, %f\n", i, h_output_X[i].x, h_output_X[i].y, h_output_X[i].z);
    }

    

    return 0;
}