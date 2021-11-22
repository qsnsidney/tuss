#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>

#include "physics.h"
#include "csv.h"
#include "helper.h"
#include "data_t.h"
#include "constant.h"
#include "basic_kernel.h"
#include <cassert>

// Comment out this line to enable debug mode
// #define NDEBUG

#define SIM_TIME 10
#define STEP_SIZE 1


int main(int argc, char *argv[])
{
    // leave this here just for simple reference 
    //CORE::VEL v{1.0f, 2.0f, 3.0f};
    //v *= 2.0f;
    //std::cout << v << std::endl;

    /* Get Dimension */
    /// TODO: Add more arguments for input and output
    /// Haiqi: I think it should be "main [num_body] [simulation_end_time] [num_iteration] or [step_size]". or we simply let step_size = 1
    if (argc != 3)
    {
        printf("Error: The number of arguments is %d, but not exactly 2\n", argc);
        return 0;
    }

    /* CSV files of initial conditions */
    unsigned nBody = atoi(argv[1]);
    std::string csv_path(argv[2]);
    // temporarily assign them to MARCO
    unsigned simulation_time = SIM_TIME;
    unsigned step_size = STEP_SIZE;
    
    /* CSV files of initial conditions */

    auto ic = CORE::parse_body_ic_from_csv(csv_path); 
    
    // TODO: get better debug message.
    assert(ic.size() == nBody);

    // random initializer just for now
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
	
    std::cout << "Start Computation\n";

    for (unsigned step = 0; step < simulation_time; step += step_size)
    {

        // There should be more than one ways to do synchronization. I temporarily randomly choosed one
        calculate_acceleration<<<nblocks, nthreads>>>(nBody, d_X[src_index], d_M,                                                          //input
                                                      d_A[dest_index]);                                                                    // output
        update_step<<<nblocks, nthreads>>>(nBody, (data_t)step_size, d_X[src_index], d_V[src_index], d_A[src_index], d_M, d_A[dest_index], //input
                                           d_X[dest_index], d_V[dest_index]);                                                              // output
	
	// we don't have to synchronize here but this gices a better visualization on how fast / slow the program is 	
	std::cout << "epoch " << step << std::endl;
	cudaDeviceSynchronize();

        swap(src_index, dest_index);
    }
    cudaDeviceSynchronize();
    std::cout << "Finished Compuation\n";
    // at end, the final data is actually at src_index because the last swap
    cudaMemcpy(h_output_X, d_X[src_index], vector_size, cudaMemcpyDeviceToHost);

    // Just for debug purpose on small inputs
    for (unsigned i = 0; i < nBody; i++)
    {
        //printf("object = %d, %f, %f, %f\n", i, h_output_X[i].x, h_output_X[i].y, h_output_X[i].z);
    }

    return 0;
}
