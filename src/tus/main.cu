#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <sys/time.h>
#include <assert.h>
#include <iostream>

#include "core/physics.hpp"
#include "core/serde.h"
#include "helper.h"
#include "data_t.h"
#include "constant.h"
#include "basic_kernel.h"
#include "core/timer.h"
#include "simple_engine.cuh"

// Comment out this line to enable debug mode
// #define NDEBUG

#define SIM_TIME 10
#define STEP_SIZE 1
#define DEFAULT_BLOCK_SIZE 32
int main(int argc, char *argv[])
{
    CORE::TIMER timer("tus");

    /* Get Dimension */
    /// TODO: Add more arguments for input and output
    /// Haiqi: I think it should be "main [num_body] [simulation_end_time] [num_iteration] or [step_size]". or we simply let step_size = 1
    if (argc < 3 or argc > 4)
    {
        printf("Error: The number of arguments must be either 3 or 4\n");
        printf("Expecting: <nbodies> <path_to_bin> <thread_per_block(optional)>\n");
        return 1;
    }
    /* BIN file of initial conditions */
    unsigned nBody = atoi(argv[1]);
    std::string bin_path(argv[2]);
    unsigned nthreads = DEFAULT_BLOCK_SIZE;
    if (argc == 4)
    {
        nthreads = atoi(argv[3]);
        // i know some none-power-of 2 also makes sense
        // but incase someone enter a dumb number, assert it here
        // later this can be removed
        assert(IsPowerOfTwo(nthreads));
    }
    timer.elapsed_previous("parsing_args");

    // temporarily assign them to MARCO
    unsigned simulation_time = SIM_TIME;
    unsigned step_size = STEP_SIZE;

    /* BIN file of initial conditions */
    auto ic = CORE::deserialize_body_state_vec_from_bin(bin_path);
    timer.elapsed_previous("loading_ic");

    // TODO: get better debug message.
    assert(ic.size() >= nBody);

    // Select engine here
    std::unique_ptr<CORE::ENGINE> engine(new TUS::SIMPLE_ENGINE(std::move(ic), step_size, nthreads));
    timer.elapsed_previous("initializing_engine");

    engine->run(simulation_time / step_size);
    timer.elapsed_previous("running_engine");

    return 0;
}
