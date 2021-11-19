cmake_minimum_required(VERSION 3.0.0)
project(cpu_sim VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(use_core_lib, 0)

if(use_core_lib)
    file(GLOB cpu_sim_SRC
        "*.h"
        "*.cpp"
    )

    add_executable(cpu_sim ${cpu_sim_SRC})
    target_include_directories(cpu_sim PRIVATE ../core/)
    target_link_libraries(cpu_sim core_lib)

    message(STATUS "cpu_sim uses core via core_lib")
else()
    file(GLOB mixed_SRC
        "*.h"
        "*.cpp"
        "../core/*.h"
        "../core/*.cpp"
    )

    add_executable(cpu_sim ${mixed_SRC})
    target_include_directories(cpu_sim PRIVATE ../core/)

    message(STATUS "cpu_sim uses core via shared files")
endif()