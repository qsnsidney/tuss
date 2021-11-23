cmake_minimum_required(VERSION 3.0.0)
project(cpusim VERSION 0.1.0)

add_compile_options(-Werror -Wall -Wno-missing-braces -O3)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR} ../core)
link_libraries(core)

file(GLOB_RECURSE cpusim_lib_SRC
    "*.cpp"
)
add_library(cpusim ${cpusim_lib_SRC})

add_executable(cpusim_exe main.cc)
target_link_libraries(cpusim_exe cpusim)

add_custom_target(cpusim_all)
add_dependencies(cpusim_all cpusim cpusim_exe)