include_directories(../)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(test_xyz test_xyz.cpp)
add_test(xyz test_xyz)