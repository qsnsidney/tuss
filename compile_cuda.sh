#!/usr/bin/env bash -x
mkdir -p build/gpu
nvcc ./$1.cu -o build/gpu/$1
