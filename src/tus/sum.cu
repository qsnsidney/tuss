#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double getTimeStamp() 
{
    struct timeval  tv ; gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

void printMatrix(float* M, char* name, int r)
{
    int i;
    printf("%s\n", name);
    for (i = 0; i < r; i++)
    {
        printf("%f ", M[i]);
    }
    printf("=============================\n");
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(int argc, char *argv[])
{
    if (argc != 2 )
    {
        printf("Error: Wrong number of arguments :(\n");
        exit(1);
    }

    int r = atoi(argv[1]);
    if (r<0)
    {
        printf("Error: Arguments not greater than 0\n");
        exit(1);
    }
    int i;
    double total_GPU_time=0, CPU_GPU_transfer_time=0, kernel_time=0, GPU_CPU_transfer_time=0, Z_value=0;
    int num = r*sizeof(float);

    // Dynamically allocate and initialize required matrices
    float* h_X, *h_dZ;
    cudaError_t err;
    err = cudaHostAlloc( (void **) &h_X, num, 0 ) ;
    if( err != cudaSuccess )
    {
        printf( "Error: %s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }
    err = cudaHostAlloc( (void **) &h_dZ, sizeof(float), 0 ) ;
    if( err != cudaSuccess )
    {
        printf( "Error: %s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }
    float h_hZ = 0;

    for (i = 0; i < r; i++)
    {
        h_X[i] = 1;
    }

    //printMatrix(h_X, "h_X", r, c);
    //printMatrix(h_Y, "h_Y", r, c);

    // CPU reference
    for (i = 0; i < r; i++)
    {
        h_hZ += h_X[i];
    }

    //printMatrix(h_hZ, "h_hZ", r, c);

    // CPU to GPU transfer
    float *d_X, *d_Z;
    err = cudaMalloc( (void **) &d_X, num ) ;
    if( err != cudaSuccess )
    {
        printf( "Error: %s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }

    double cg_transfer_start = getTimeStamp();
    err = cudaMemcpy( d_X, h_X, num, cudaMemcpyHostToDevice ) ;
    if( err != cudaSuccess )
    {
        printf( "Error: %s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }
    double cg_transfer_end = getTimeStamp();

    // Invoke kernel
    double kernel_start = getTimeStamp();
    dim3 block = 32;
    int blockNum = (r + block.x-1)/block.x;
    err = cudaMalloc( (void **) &d_Z, blockNum*sizeof(float) ) ;
    if( err != cudaSuccess )
    {
        printf( "Error: %s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }

    const int bs = 32;
    reduce<<<blockNum, block>>><bs>( d_X, d_Z, r) ;
    while (blockNum != 1)
    {
        int total = blockNum;
        blockNum = (blockNum + block.x-1)/block.x;
        reduce<<<blockNum, block>>><bs>( d_X, d_Z, total) ;
    }

    cudaDeviceSynchronize();
    double kernel_end = getTimeStamp();

    // GPU to CPU transfer
    double gc_transfer_start = getTimeStamp();
    err = cudaMemcpy( h_dZ, d_Z[0], sizeof(float), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess )
    {
        printf( "Error: %s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }
    double gc_transfer_end = getTimeStamp();

    printMatrix(h_dZ, "h_dZ", r);

    //Report: <total_GPU_time> <CPU_GPU_transfer_time> <kernel_time> <GPU_CPU_transfer_time> <Z-value>
    CPU_GPU_transfer_time = cg_transfer_end - cg_transfer_start;
    kernel_time = kernel_end - kernel_start;
    GPU_CPU_transfer_time = gc_transfer_end - gc_transfer_start;
    total_GPU_time = gc_transfer_end - cg_transfer_start;
    if (r > 5 && c > 5) Z_value = h_dZ[5*c+5];
    printf("<%f> <%f> <%f> <%f> <%f>\n", total_GPU_time, CPU_GPU_transfer_time, kernel_time, GPU_CPU_transfer_time, Z_value);

    // GPU & CPU result comparison
    if (h_hZ != h_dZ)
    {
        printf("Error: GPU result is different from CPU - kernel has errors!\n");
    }

    // Free resources
    cudaFree(d_X);
    cudaFree(d_Z);
    cudaFreeHost(h_X);
    cudaFreeHost(h_dZ);
    cudaDeviceReset();
}
