include_directories(../)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(test_xyz test_xyz.cpp)
add_test(test_xyz test_xyz)

add_executable(test_csv test_csv.cpp)
target_link_libraries(test_csv core_lib)
add_test(test_csv test_csv)