#!/usr/bin/env python3

import os
import re
import subprocess
import argparse

def error_and_exit(err_msg):
    print(err_msg)
    exit(1)


if __name__=='__main__':
    THREAD_PER_BLOCK = [16,32,64,128,256]
    NBODY = [10000,20000,50000,100000]
    AVG_ITERATION = 1
    CUDA_EXECUTABLE = "build/tus/tus_exe"
    GPU_TIME_PATTERN = "Profile \[computation_core\] took (([0-9]*[.])?[0-9]+)"
    BENCHMARK_DATA = "benchmark/ic/benchmark_500000.bin"
    BENCHMARK_OUTPUT_FILE = "gpu_benchmark.csv"
    STDOUT_OUTPUT = "benchmark.stdout"

    parser = argparse.ArgumentParser(description='Simple parser')
    parser.add_argument('--iter', type=int, default=AVG_ITERATION,
                        help='number of runs for each configuration (default: 1)')

    args = parser.parse_args()

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

    for block_size in THREAD_PER_BLOCK:
        f_data.write("\n")
        f_data.write(str(block_size) + ",")
        for num_body in NBODY:
            total_time = 0
            for count in range(args.iter):
                info_msg = "RUNNING NUMBLOCK : " + str(block_size) + ". NBODY : " + str(num_body) + ". ITER: " + str(count)
                f_stdout.write(info_msg + "\n")
                print(info_msg)
                command = [cuda_executable, '-b ' + str(num_body), '-i ' + benchmark_path, '-t ' + str(block_size), '-d 1', '-n 10']
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
                total_time += float(gpu_runtime)
            avg_time = total_time / AVG_ITERATION
            f_data.write("{:.6f}".format(avg_time) + ",")

    f_data.close()
    f_stdout.close()