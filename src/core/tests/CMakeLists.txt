cmake_minimum_required(VERSION 3.7.0)
project(core_tests)

include_directories(..)
link_libraries(core)

add_compile_options(-Werror -Wall -Wno-missing-braces -O3 -march=native)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(xyz_tests xyz_tests.cpp)
add_test(core_tests_xyz xyz_tests)

add_executable(serde_tests serde_tests.cpp)
add_test(core_tests_serde serde_tests)

add_executable(physics_tests physics_tests.cpp)
add_test(core_tests_physics physics_tests)

# Add test executable here
add_custom_target(core_tests)
add_dependencies(core_tests xyz_tests serde_tests physics_tests)