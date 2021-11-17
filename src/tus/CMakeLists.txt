cmake_minimum_required(VERSION 3.9.0)
project(tus VERSION 0.1.0 LANGUAGES C CXX CUDA)

cuda_add_executable(tus main.cpp)