#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Comment out this line to enable debug mode
// #define NDEBUG

#define RANDOM_RANGE 5
#define EPSILON 0.01

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
// WARNING: this function has hardcoded assumption on float vs double
__host__ __device__ inline data_t_3d make_data_t_3d(const data_t a, const data_t b, const data_t c) {
    return make_float3(a, b, c);
}

__host__ __device__ data_t_3d operator+(const data_t_3d & a, const data_t_3d & b) {
  
  return make_data_t_3d(a.x+b.x, a.y+b.y, a.z+b.z);

}

__host__ __device__ data_t_3d operator-(const data_t_3d & a, const data_t_3d & b) {
  
  return make_data_t_3d(a.x-b.x, a.y-b.y, a.z-b.z);

}

__host__ __device__ data_t_3d operator*(const data_t_3d & a, const data_t & b) {
  
  return make_data_t_3d(a.x * b, a.y * b, a.z * b);

}

__host__ __device__ data_t_3d operator/(const data_t_3d & a, const data_t & b) {
  
  return make_data_t_3d(a.x / b, a.y / b, a.z / b);

}

__host__ data_t gen_random_data_t(unsigned upper_bound) {
    return (data_t)rand()/(data_t)(RAND_MAX/(2 * upper_bound)) - (data_t)upper_bound;
}

// randomly initialize the input array with type data_t in [-range, range)
__host__ void random_initialize_vector(data_t_3d * input_array, size_t size, data_t range) {
    for(size_t i = 0; i < size; i++){
        //input_array[i] = make_data_t_3d(gen_random_data_t(range), gen_random_data_t(range), gen_random_data_t(range));
        input_array[i] = make_data_t_3d(i, 2 * i, 3 * i);
    }        
}

__host__ void random_initialize_mass(data_t * input_array, size_t size, data_t range){
    //hack: + range at the end to offset the nagative
    for(size_t i = 0; i < size; i++){
        //input_array[i] = gen_random_data_t(range) + range;
        input_array[i] = (float)(i) / 2;
    }
}

// WARNING: this function has hardcoded assumption on float vs double
__device__ data_t power_norm(data_t_3d a, data_t_3d b){
    return powf(powf(a.x - b.x,2) + powf(a.y - b.y,2) + powf(a.z - b.z,2) + powf(EPSILON,2), 1.5);
}

/*
 * Here starts the actual kernel implemetation
 */

// temporary throw a dummy kernel just for very basic level sanity check
__global__ void kernel_place_holder(data_t_3d * input_ptr, data_t_3d * input_ptr2, data_t_3d * output_ptr, int nsize) {
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nsize){
        output_ptr[tid] = input_ptr2[tid] + input_ptr[tid];
    }
}

__global__ void calculate_acceleration(data_t_3d * location, data_t * mass, data_t_3d* acceleration, int nbody) {
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if(tid < nbody){
        data_t_3d accumulated_accer = make_data_t_3d(0, 0, 0);
        data_t_3d x_self = location[tid];
        for(unsigned j = 0; j < nbody; j++){
            if(j == tid) {
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
    data_t * d_M;
    d_X = (data_t_3d**)malloc(2 * sizeof(data_t_3d*));
    gpuErrchk(cudaMalloc((void**)&d_X[0], vector_size));
    gpuErrchk(cudaMalloc((void**)&d_X[1], vector_size));

    d_A = (data_t_3d**)malloc(2 * sizeof(data_t_3d*));
    gpuErrchk(cudaMalloc((void**)&d_A[0], vector_size));
    gpuErrchk(cudaMalloc((void**)&d_A[1], vector_size));    
    
    d_V = (data_t_3d**)malloc(2 * sizeof(data_t_3d*));
    gpuErrchk(cudaMalloc((void**)&d_V[0], vector_size));
    gpuErrchk(cudaMalloc((void**)&d_V[1], vector_size));

    gpuErrchk(cudaMalloc((void**)&d_M, data_size));

    /*
     *   create double buffer on device side
     */ 
    // cudaMemcpy(d_A[0], h_A, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X[0], h_X, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V[0], h_V, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, data_size, cudaMemcpyHostToDevice);

    int nblocks = 4, nthreads=4;
    calculate_acceleration<<<nblocks, nthreads>>>(d_X[0], d_M, d_A[0], nBody);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output_X, d_A[0], vector_size, cudaMemcpyDeviceToHost);

    // Just for debug purpose on small inputs
    for(int i = 0; i < nBody; i++){
        printf("object = %d, %f, %f, %f\n", i, h_output_X[i].x, h_output_X[i].y, h_output_X[i].z);
    }

    //for(int i = 0; i < nBody; i++){
    //    printf("locations: tid = %d, %f, %f, %f\n", i, h_X[i].x, h_X[i].y, h_X[i].z);
    //    printf("mass: tid = %d, %f\n", i, h_M[i]);
    //}
    

    return 0;
}