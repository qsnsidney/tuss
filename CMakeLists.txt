cmake_minimum_required(VERSION 3.9.0)
project(ECE1782)

include(CTest)

add_subdirectory(src/core core)
add_subdirectory(src/cpu_sim cpu_sim)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(src/tus tus)
else()
    message(WARNING "CUDA is not found! Skipping tus.")
endif()