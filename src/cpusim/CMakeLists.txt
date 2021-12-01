cmake_minimum_required(VERSION 3.0.0)
project(cpusim VERSION 0.1.0)

find_package(Threads REQUIRED)

add_compile_options(-Werror -Wall -Wno-missing-braces -O3)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options(-march=native)
    message(STATUS "-march=native is enabled for cpusim and cpusim_exe")
endif()
if(ENABLE_FFAST_MATH)
    add_compile_options(-ffast-math)
    message(STATUS "-ffast-math is enabled for cpusim and cpusim_exe")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR} ../)
link_libraries(core)

file(GLOB_RECURSE cpusim_lib_SRC
    "*.cpp"
)
add_library(cpusim ${cpusim_lib_SRC})
target_link_libraries(cpusim PRIVATE Threads::Threads)

add_executable(cpusim_exe main.cc)
target_link_libraries(cpusim_exe cpusim)

add_custom_target(cpusim_all)
add_dependencies(cpusim_all cpusim cpusim_exe)