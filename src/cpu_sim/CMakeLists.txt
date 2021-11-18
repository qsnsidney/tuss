cmake_minimum_required(VERSION 3.0.0)
project(cpu_sim VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB SOURCES
    ./*.h
    ./*.cpp
    ../core/*.h
    ../core/*.cpp
)

include_directories(../core/)
add_executable(cpu_sim ${SOURCES})