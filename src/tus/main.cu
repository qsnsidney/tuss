#include <stdlib.h>
#include <math.h>
#include <memory>
#include <iostream>

#include "core/physics.hpp"
#include "core/serde.h"
#include "helper.cuh"
#include "core/timer.h"
#include "core/macros.hpp"
#include "core/utility.hpp"
#include "simple_engine.cuh"
#include "nvda_reference_engine.cuh"
#include "small_N_engine.cuh"
#include "coalesced_simple_engine.cuh"
#include "tiled_simple_engine.cuh"
#include "mat_mul_engine.cuh"
#include "nvda_improved_engine.cuh"
#include "core/cxxopts.hpp"
#include "cpusim/reference.h"

namespace
{
    constexpr size_t default_block_size = 32;

    enum class VERSION
    {
        BASIC = 0,
        NVDA_REFERENCE,
        COALESCED_BASIC,
        TILED_BASIC,
        SMALL_N,
        MAT_MUL,
        NVDA_IMPROVED,
    };
}

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
    option_group("t,block_size", "num_threads_per_block for CUDA", cxxopts::value<int>()->default_value(std::to_string(::default_block_size)));
    option_group("len", "horizontal dimension of a 2D thread block", cxxopts::value<int>()->default_value("16"));
    option_group("wid", "vertical dimension of a 2D thread block", cxxopts::value<int>()->default_value("32"));
    option_group("lur", "loop unrolling factor", cxxopts::value<int>()->default_value("1"));
    option_group("tpb", "thread per body", cxxopts::value<int>()->default_value("32"));
    option_group("V,version", "version of optimization (see src/tus/main.cu): optional (default 0)",
                 cxxopts::value<int>()->default_value(std::to_string(static_cast<int>(VERSION::BASIC))));
    option_group("o,out", "system_state_log_dir: optional", cxxopts::value<std::string>());
    option_group("snapshot", "only dump out the final view, combined with --out: optional (default false)");
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
    const VERSION version = static_cast<VERSION>(arg_result["version"].as<int>());
    std::optional<std::string> system_state_log_dir_opt = {};
    if (arg_result.count("out"))
    {
        system_state_log_dir_opt = arg_result["out"].as<std::string>();
    }
    const bool snapshot = static_cast<bool>(arg_result.count("snapshot"));
    const bool verify = static_cast<bool>(arg_result.count("verify"));
    const int verbosity = arg_result.count("verbose");
    CORE::TIMER::set_trigger_level(static_cast<CORE::TIMER::TRIGGER_LEVEL>(verbosity));

    const int tb_len = arg_result["len"].as<int>();
    const int tb_wid = arg_result["wid"].as<int>();
    const int unroll_factor = arg_result["lur"].as<int>();
    const int tpb = arg_result["tpb"].as<int>();

    std::cout << "Running.." << std::endl;
    std::cout << "ic_file: " << ic_file_path << std::endl;
    std::cout << "max_n_body: " << max_n_body << std::endl;
    std::cout << "dt: " << dt << std::endl;
    std::cout << "n_iteration: " << n_iteration << std::endl;
    std::cout << "block_size: " << block_size << std::endl;
    std::cout << "version: " << static_cast<int>(version) << std::endl;
    std::cout << "system_state_log_dir: " << (system_state_log_dir_opt ? *system_state_log_dir_opt : std::string("null")) << std::endl;
    std::cout << "snapshot: " << snapshot << std::endl;
    std::cout << "verify: " << verify << std::endl;
    std::cout << "verbosity: " << verbosity << std::endl;
    std::cout << "len: " << tb_len << std::endl;
    std::cout << "wid: " << tb_wid << std::endl;
    std::cout << "lur: " << unroll_factor << std::endl;
    std::cout << "tpb: " << tpb << std::endl;
    std::cout << std::endl;
    timer.elapsed_previous("parsing_args");

    // Verification flow setting
    if (verify)
    {
        std::cout << "--------------------" << std::endl;
        std::cout << "VERIFICATION FLOW" << std::endl;
        std::cout << "OVERRIDE:" << std::endl;
        std::cout << "--------------------" << std::endl;
    }

    /* BIN file of initial conditions */
    CORE::SYSTEM_STATE
        system_state_ic = CORE::deserialize_system_state_from_file(ic_file_path);
    if (max_n_body >= 0 && max_n_body < static_cast<int>(system_state_ic.size()))
    {
        system_state_ic.resize(max_n_body);
        std::cout << "Limiting number of bodies to " << max_n_body << std::endl;
    }
    timer.elapsed_previous("loading_ic");

    // Select engine here
    const std::optional<std::string> system_state_engine_log_dir_opt = snapshot ? std::nullopt : system_state_log_dir_opt;
    std::unique_ptr<CORE::ENGINE> engine;
    if (version == VERSION::NVDA_REFERENCE)
    {
        engine = std::make_unique<TUS::NVDA_REFERENCE_ENGINE>(
            system_state_ic, dt, block_size, system_state_engine_log_dir_opt);
    }
    else if (version == VERSION::BASIC)
    {
        engine = std::make_unique<TUS::SIMPLE_ENGINE>(
            system_state_ic, dt, block_size, system_state_engine_log_dir_opt);
    }
    else if (version == VERSION::COALESCED_BASIC)
    {
        engine = std::make_unique<TUS::COALESCED_SIMPLE_ENGINE>(
            system_state_ic, dt, block_size, system_state_engine_log_dir_opt);
    }
    else if (version == VERSION::TILED_BASIC)
    {
        engine = std::make_unique<TUS::TILED_SIMPLE_ENGINE>(
            system_state_ic, dt, block_size, system_state_engine_log_dir_opt);
    }
    else if (version == VERSION::MAT_MUL)
    {
        // POC, bad performance due to too many kernel calls
        engine = std::make_unique<TUS::MAT_MUL_ENGINE>(
            system_state_ic, dt, block_size, system_state_engine_log_dir_opt);
    }
    else if (version == VERSION::NVDA_IMPROVED)
    {
        engine = std::make_unique<TUS::NVDA_IMPROVED_ENGINE>(
            system_state_ic, dt, block_size, system_state_engine_log_dir_opt);
    }
    else if (version == VERSION::SMALL_N) 
    {
            engine.reset(new TUS::SMALL_N_ENGINE(
            system_state_ic, dt, block_size, tb_len,tb_wid, unroll_factor, tpb, system_state_engine_log_dir_opt));
    }
    else 
    {
        std::cout << "INVALID ENGINE VALUE: " << static_cast<std::underlying_type<VERSION>::type>(version) << std::endl;
        exit(1);
    }
    timer.elapsed_previous("initializing_engine");

    // Execute engine
    const CORE::SYSTEM_STATE &actual_system_state_result = engine->run(n_iteration);
    timer.elapsed_previous("running_engine");

    if (snapshot && system_state_log_dir_opt)
    {
        const std::string delim = "/";
        const std::string snapshot_filename =
            *system_state_log_dir_opt + delim + CORE::remove_extension(CORE::base_name(ic_file_path)) +
            "_" + std::to_string(static_cast<size_t>(dt * n_iteration)) + ".bin";
        CORE::serialize_system_state_to_bin(snapshot_filename, actual_system_state_result, true);
    }

    if (verify)
    {
        std::cout << "====================" << std::endl;
        std::cout << "VERIFYING.." << std::endl;
        const bool result = CPUSIM::run_verify_with_reference_engine(system_state_ic, actual_system_state_result, dt, n_iteration);
        std::cout << "VERFICATION RESULT:" << std::endl;
        if (result)
        {
            std::cout << "    SUCCESSFUL" << std::endl;
        }
        else
        {
            std::cout << "    FAILED" << std::endl;
        }
        std::cout << "====================" << std::endl;
    }
    timer.elapsed_previous("verify");

    return 0;
}
