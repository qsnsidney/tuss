#!/usr/bin/env python3

import os
import re
import subprocess

def error_and_exit(err_msg):
    print(err_msg)
    exit(1)

THREAD_PER_BLOCK = [16,32,64,128,256]
NBODY = [10000,20000,50000,100000,200000]
CUDA_EXECUTABLE = "build/tus/tus_exe"
GPU_TIME_PATTERN = "Subprofile \(Finished computation\) took (([0-9]*[.])?[0-9]+)"
BENCHMARK_DATA = "benchmark/benchmark_500000.ic.bin"
BENCHMARK_OUTPUT_FILE = "gpu_benchmark.csv"
STDOUT_OUTPUT = "benchmark.stdout"

script_dir = os.path.dirname(os.path.realpath(__file__))
project_home_dir = os.path.join(script_dir, "../../")

data_output_file_path = os.path.join(project_home_dir, BENCHMARK_OUTPUT_FILE)
stdout_file_path = os.path.join(project_home_dir, STDOUT_OUTPUT)
benchmark_path = os.path.join(project_home_dir, BENCHMARK_DATA)
cuda_executable = os.path.join(project_home_dir, CUDA_EXECUTABLE)

f_data = open(data_output_file_path, "w")
f_stdout = open(stdout_file_path, "w")

f_data.write("nbody/block")
for i in NBODY:
    f_data.write("," + str(i))

for num_block in THREAD_PER_BLOCK:
    f_data.write("\n")
    f_data.write(str(num_block) + ",")
    for num_body in NBODY:

        f_stdout.write("NUMBLOCK : " + str(num_block) + ". NBODY : " + str(num_body) + "\n")

        command = [cuda_executable, str(num_body), benchmark_path, str(num_block)]
        try:
            result = subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(command)
            error_and_exit(e.output.decode('utf-8'))

        ret = result.decode('utf-8')
        f_stdout.write(ret + "\n") 
        gpu_runtime_re = re.search(GPU_TIME_PATTERN, ret)
        if not gpu_runtime_re:
            error_and_exit("failed to find gpu runtime")
        gpu_runtime = gpu_runtime_re.group(1)
        f_data.write(gpu_runtime + ",")

f_data.close()
f_stdout.close()