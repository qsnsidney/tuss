#include <iostream>
#include <memory>
#include <optional>

#include "core/macros.hpp"
#include "core/serde.h"
#include "core/engine.h"
#include "core/timer.h"
#include "core/cxxopts.hpp"
#include "core/utility.hpp"
#include "basic_engine.h"
#include "shared_acc_engine.h"
#include "reference.h"

namespace
{
    enum class VERSION
    {
        BASIC = 0,
        SHARED_ACC
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
    option_group("t,num_threads", "num_threads for CPU", cxxopts::value<int>()->default_value("1"));
    option_group("thread_pool", "use thread pool for multithreading: optional (default off)");
    option_group("V,version", "version of optimization (0 - basic, 1 - shared acc edge): optional (default 1)",
                 cxxopts::value<int>()->default_value(std::to_string(static_cast<int>(VERSION::SHARED_ACC))));
    option_group("o,out", "system_state_log_dir: optional (default null)", cxxopts::value<std::string>());
    option_group("snapshot", "only dump out the final view, combined with --out: optional (default false)");
    option_group("verify", "verify 1 iteration result with reference algorithm: optional (default off)");
    option_group("v,verbose", "verbosity: can stack, optional (default off)");
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
    CORE::TIMER timer("cpusim");

    // Load args
    auto arg_result = parse_args(argc, argv);
    const std::string ic_file_path = arg_result["ic_file"].as<std::string>();
    const int max_n_body = arg_result["num_bodies"].as<int>();
    const CORE::DT dt = arg_result["dt"].as<CORE::UNIVERSE::floating_value_type>();
    const int n_iteration = arg_result["num_iterations"].as<int>();
    const int n_thread = arg_result["num_threads"].as<int>();
    const bool use_thread_pool = static_cast<bool>(arg_result.count("thread_pool"));
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

    std::cout << "Running.." << std::endl;
    std::cout << "ic_file: " << ic_file_path << std::endl;
    std::cout << "max_n_body: " << max_n_body << std::endl;
    std::cout << "dt: " << dt << std::endl;
    std::cout << "n_iteration: " << n_iteration << std::endl;
    std::cout << "n_thread: " << n_thread << std::endl;
    std::cout << "use_thread_pool: " << use_thread_pool << std::endl;
    std::cout << "version: " << static_cast<int>(version) << std::endl;
    std::cout << "system_state_log_dir: " << (system_state_log_dir_opt ? *system_state_log_dir_opt : std::string("null")) << std::endl;
    std::cout << "snapshot: " << snapshot << std::endl;
    std::cout << "verify: " << verify << std::endl;
    std::cout << "verbosity: " << verbosity << std::endl;
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

    // Load ic
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
    if (version == VERSION::SHARED_ACC)
    {
        engine.reset(new CPUSIM::SHARED_ACC_ENGINE(
            system_state_ic, dt, n_thread, use_thread_pool, system_state_engine_log_dir_opt));
    }
    else
    {
        engine.reset(new CPUSIM::BASIC_ENGINE(
            system_state_ic, dt, n_thread, use_thread_pool, system_state_engine_log_dir_opt));
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