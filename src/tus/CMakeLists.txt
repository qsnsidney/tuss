cmake_minimum_required(VERSION 3.9.0)
project(tus VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE tus_exe_SRC
    "*.cu",
    "*.cpp"
)

add_executable(tus_exe ${tus_exe_SRC})
target_include_directories(tus_exe PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(tus_exe PRIVATE ../)
target_link_libraries(tus_exe core cpusim)

# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED True)
# Work around issues for CMake older than ver 3.18
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")
target_compile_options(tus_exe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --std=c++17
                       -res-usage
                       # --relocatable-device-code=true
                       >)