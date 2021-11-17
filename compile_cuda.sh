#!/usr/bin/env bash
mkdir -p build
cd build 
nvcc ../$1.cu -o $1
cd ..
