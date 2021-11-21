cmake_minimum_required(VERSION 3.0.0)
project(cpu_sim VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR} ../core)
link_libraries(core)

file(GLOB_RECURSE cpu_sim_lib_SRC
    "*.cpp"
)
add_library(cpu_sim ${cpu_sim_lib_SRC})

add_executable(cpu_sim_exe main.cc)
target_link_libraries(cpu_sim_exe cpu_sim)

add_custom_target(cpu_sim_all)
add_dependencies(cpu_sim_all cpu_sim cpu_sim_exe)