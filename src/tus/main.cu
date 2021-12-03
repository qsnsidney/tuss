#include <stdlib.h>
#include <math.h>
#include <memory>
#include <iostream>

#include "core/physics.hpp"
#include "core/serde.h"
#include "helper.cuh"
#include "constant.h"
#include "core/timer.h"
#include "core/macros.hpp"
#include "simple_engine.cuh"
#include "core/cxxopts.hpp"

#define DEFAULT_BLOCK_SIZE 32

auto parse_args(int argc, const char *argv[])
{
    cxxopts::Options options(argv[0]);
    options
        .positional_help("[optional args]")
        .show_positional_help()
        .set_tab_expansion()
        .allow_unrecognised_options();

    auto option_group = options.add_options();
    option_group("i,ic_file", "ic_file: .bin or .csv", cxxopts::value<std::string>());
    option_group("b,num_bodies", "max_n_bodies: optional (default -1), no effect if < 0 or >= n_body from ic_file", cxxopts::value<int>()->default_value("-1"));
    option_group("d,dt", "dt", cxxopts::value<CORE::UNIVERSE::floating_value_type>());
    option_group("n,num_iterations", "num_iterations", cxxopts::value<int>());
    option_group("t,block_size", "num_threads_per_block for CUDA", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_BLOCK_SIZE)));
    option_group("o,out", "system_state_log_dir: optional", cxxopts::value<std::string>());
    option_group("verify", "verify 1 iteration result with reference algorithm: optional (default off)");
    option_group("v,verbose", "verbosity: can stack, optional");
    option_group("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    return result;
}

int main(int argc, const char *argv[])
{
    CORE::TIMER timer("tus");

    // Load args
    auto arg_result = parse_args(argc, argv);
    const std::string ic_file_path = arg_result["ic_file"].as<std::string>();
    const int max_n_body = arg_result["num_bodies"].as<int>();
    const CORE::DT dt = arg_result["dt"].as<CORE::UNIVERSE::floating_value_type>();
    const int n_iteration = arg_result["num_iterations"].as<int>();
    const int block_size = arg_result["block_size"].as<int>();
    // I know some none-power-of 2 also makes sense
    // but incase someone enter a dumb number, assert it here
    // later this can be removed
    ASSERT(IsPowerOfTwo(block_size));
    std::optional<std::string> system_state_log_dir_opt = {};
    if (arg_result.count("out"))
    {
        system_state_log_dir_opt = arg_result["out"].as<std::string>();
    }
    const bool verify = static_cast<bool>(arg_result.count("verify"));
    const int verbosity = arg_result.count("verbose");
    CORE::TIMER::set_trigger_level(static_cast<CORE::TIMER::TRIGGER_LEVEL>(verbosity));

    std::cout << "Running.." << std::endl;
    std::cout << "ic_file: " << ic_file_path << std::endl;
    std::cout << "max_n_body: " << max_n_body << std::endl;
    std::cout << "dt: " << dt << std::endl;
    std::cout << "n_iteration: " << n_iteration << std::endl;
    std::cout << "block_size: " << block_size << std::endl;
    std::cout << "system_state_log_dir: " << (system_state_log_dir_opt ? *system_state_log_dir_opt : std::string("null")) << std::endl;
    std::cout << "verify: " << verify << std::endl;
    std::cout << "verbosity: " << verbosity << std::endl;
    std::cout << std::endl;
    timer.elapsed_previous("parsing_args");

    /* BIN file of initial conditions */
    CORE::SYSTEM_STATE
        system_state = CORE::deserialize_system_state_from_file(ic_file_path);
    if (max_n_body >= 0 && max_n_body < static_cast<int>(system_state.size()))
    {
        system_state.resize(max_n_body);
        std::cout << "Limiting number of bodies to " << max_n_body << std::endl;
    }
    timer.elapsed_previous("loading_ic");

    // Select engine here
    std::unique_ptr<CORE::ENGINE> engine(new TUS::SIMPLE_ENGINE(std::move(system_state), dt, block_size, system_state_log_dir_opt));
    timer.elapsed_previous("initializing_engine");

    const int n_iteration_to_run = verify ? 1 : n_iteration;
    std::cout << "INFO: Ready to run " << n_iteration_to_run << " iterations" << std::endl;

    // Execute engine
    engine->run(n_iteration_to_run);
    timer.elapsed_previous("running_engine");

    return 0;
}
