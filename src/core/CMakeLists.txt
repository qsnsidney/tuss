cmake_minimum_required(VERSION 3.0.0)
project(core VERSION 0.1.0)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB core_SRC
     "*.cpp"
)
add_library(core_lib ${core_SRC})
target_include_directories(core_lib PUBLIC ${CMAKE_SOURCE_DIR})

add_subdirectory(tests)