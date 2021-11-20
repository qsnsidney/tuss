cmake_minimum_required(VERSION 3.0.0)
project(cpu_sim VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE cpu_sim_SRC
    "*.cpp"
)

add_executable(cpu_sim ${cpu_sim_SRC})
target_include_directories(cpu_sim PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(cpu_sim PRIVATE ../core/)
target_link_libraries(cpu_sim core_lib)
