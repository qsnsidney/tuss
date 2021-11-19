cmake_minimum_required(VERSION 3.7.0)
project(core_tests)

include_directories(../)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(xyz_tests xyz_tests.cpp)
add_test(core_tests_xyz xyz_tests)

add_executable(csv_tests csv_tests.cpp)
target_link_libraries(csv_tests core_lib)
add_test(core_tests_csv csv_tests)

add_executable(physics_tests physics_tests.cpp)
target_link_libraries(physics_tests core_lib)
add_test(core_tests_physics physics_tests)

# Add test executable here
add_custom_target(core_tests)
add_dependencies(core_tests xyz_tests csv_tests physics_tests)