#include <iostream>
#include <memory>

#include "macros.h"
#include "serde.h"
#include "engine.h"
#include "simple_engine.h"
#include "timer.h"

int main(int argc, char *argv[])
{
    CORE::TIMER timer("cpusim");

    // Load args
    ASSERT(argc == 4 && "Expect arguments: [ic_bin_file] [dt] [n_iteration]");
    const std::string ic_bin_file_path = argv[1];
    const CORE::DT dt = std::stod(argv[2]);
    const int n_iteration = std::stoi(argv[3]);
    std::cout << "Running.." << std::endl;
    std::cout << "IC: " << ic_bin_file_path << std::endl;
    std::cout << "dt: " << dt << std::endl;
    std::cout << "n_iteration: " << n_iteration << std::endl;
    std::cout << std::endl;
    timer.elapsed_previous("parsing_args");

    // Load ic
    std::vector<CORE::BODY_IC>
        body_ics = CORE::deserialize_body_ic_from_bin(ic_bin_file_path);
    timer.elapsed_previous("loading_ic");

    // Select engine here
    std::unique_ptr<CPUSIM::ENGINE> engine(new CPUSIM::SIMPLE_ENGINE);
    engine->init(std::move(body_ics));
    timer.elapsed_previous("initializing_engine");

    // Execute engine
    engine->execute(dt, n_iteration);
    timer.elapsed_previous("running_engine");

    return 0;
}