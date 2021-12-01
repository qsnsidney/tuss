cmake_minimum_required(VERSION 3.7.0)
project(core_tests)

include_directories(..)
link_libraries(core)

add_compile_options(-Werror -Wall -Wno-missing-braces -O3)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options(-march=native)
    message(STATUS "-march=native is enabled for core_tests")
endif()
if(ENABLE_FFAST_MATH)
    add_compile_options(-ffast-math)
    message(STATUS "-ffast-math is enabled for core_tests")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(xyz_tests xyz_tests.cc)
add_test(core_tests_xyz xyz_tests)

add_executable(serde_tests serde_tests.cc)
add_test(core_tests_serde serde_tests)

add_executable(physics_tests physics_tests.cc)
add_test(core_tests_physics physics_tests)

add_executable(utility_tests utility_tests.cc)
add_test(core_tests_utility utility_tests)

# Add test executable here
add_custom_target(core_tests)
add_dependencies(core_tests xyz_tests serde_tests physics_tests utility_tests)