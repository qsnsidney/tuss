cmake_minimum_required(VERSION 3.9.0)
project(tus VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

file(GLOB tus_SRC
    "*.h"
    "*.cu"
)

add_executable(tus ${tus_SRC})
target_include_directories(tus PRIVATE ../core/)
target_link_libraries(tus core_lib)