cmake_minimum_required(VERSION 3.11.0)
project(ECE1782)

include(CTest)

add_subdirectory(src/core core)
add_subdirectory(src/cpu_sim cpu_sim)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(src/tus tus)
else()
    message(WARNING "CUDA is not found! Skipping tus")
endif()