cmake_minimum_required(VERSION 3.11.0)
project(tuss)

include(CTest)

option(ENABLE_FFAST_MATH "Enable -ffast-math" OFF)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)

add_subdirectory(src/core core)
add_subdirectory(src/cpusim cpusim)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(src/tus tus)
    message(STATUS "tus is enabled since CUDA is supported")
else()
    message(WARNING "CUDA is not found! Skipping tus")
endif()