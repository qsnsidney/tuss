#!/usr/bin/env python3

import os
import re
import subprocess
import argparse
from enum import Enum


def error_and_exit(err_msg):
    print(err_msg)
    exit(1)


BASIC = 0
NVDA_REFERENCE = 1
COALESCED_BASIC = 2
TILED_BASIC = 3
SMALL_N = 4
MAT_MUL = 5
NVDA_IMPROVED = 6

TEST_ENGINES_NAME = {
    BASIC: 'basic_engine',
    NVDA_REFERENCE: 'nvda_reference_engine',
    COALESCED_BASIC: 'coalesced_basic_engine',
    TILED_BASIC: 'tiled_basic_engine',
    SMALL_N: 'small_n',
    MAT_MUL: 'matmul_engine',
    NVDA_IMPROVED: 'improved_nvda_engine',
}

DEFAULT_ITERATION = 1


def init(parser):
    parser.add_argument('--iter', type=int, default=DEFAULT_ITERATION,
                        help=f'number of runs for each configuration (default: {DEFAULT_ITERATION})')
    parser.add_argument('--version', type=int, default=NVDA_IMPROVED,
                        help=f'version of enginer to test. default = {NVDA_IMPROVED}({TEST_ENGINES_NAME[NVDA_IMPROVED]})')


def main(args):
    THREAD_PER_BLOCK = [16, 64, 256]
    NBODY = [50000, 100000]
    CUDA_EXECUTABLE = 'build/tus/tus_exe'
    GPU_TIME_PATTERN = 'Profile \[all_iters\]: (([0-9]*[.])?[0-9]+)'
    BENCHMARK_DATA = 'data/ic/s0_s112500_g100000_d100000.bin'
    STDOUT_OUTPUT = 'benchmark.stdout'
    VERSION = args.version
    AVG_ITERATION = args.iter
    BENCHMARK_OUTPUT_FILE = f'gpu_benchmark_{TEST_ENGINES_NAME[VERSION]}.csv'

    print(f'running benchmark for {TEST_ENGINES_NAME[VERSION]}')

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_home_dir = os.path.join(script_dir, '../../')

    data_output_file_path = os.path.join(
        project_home_dir, BENCHMARK_OUTPUT_FILE)
    stdout_file_path = os.path.join(project_home_dir, STDOUT_OUTPUT)
    benchmark_path = os.path.join(project_home_dir, BENCHMARK_DATA)
    cuda_executable = os.path.join(project_home_dir, CUDA_EXECUTABLE)

    f_data = open(data_output_file_path, 'w')
    f_stdout = open(stdout_file_path, 'w')

    f_data.write('nbody/block')
    for i in NBODY:
        f_data.write(',' + str(i))

    for block_size in THREAD_PER_BLOCK:
        f_data.write('\n')
        f_data.write(str(block_size) + ',')
        for num_body in NBODY:
            total_time = 0
            for count in range(AVG_ITERATION):
                info_msg = 'RUNNING NUMBLOCK : ' + \
                    str(block_size) + '. NBODY : ' + \
                    str(num_body) + '. ITER: ' + str(count)
                f_stdout.write(info_msg + '\n')
                print(info_msg)
                command = [cuda_executable, '-b', str(num_body), '-i', benchmark_path, '-t', str(
                    block_size), '-d', '1', '-n', '10', '--version', str(VERSION)]
                try:
                    result = subprocess.check_output(
                        command, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    print(command)
                    error_and_exit(e.output.decode('utf-8'))

                ret = result.decode('utf-8')
                f_stdout.write(ret + '\n')
                gpu_runtime_re = re.search(GPU_TIME_PATTERN, ret)
                if not gpu_runtime_re:
                    error_and_exit('failed to find gpu runtime')
                gpu_runtime = gpu_runtime_re.group(1)
                total_time += float(gpu_runtime)
            avg_time = total_time / AVG_ITERATION
            f_data.write('{:.6f}'.format(avg_time) + ',')

    f_data.close()
    f_stdout.close()


if __name__ == '__main__':
    print('WARNING:', 'Launching directly from this script is now deprecated!')
    print('WARNING:', 'Please use:')
    print('WARNING:', '  python3 -m scripts.benchmark cuda')
    print(' ')
    parser = argparse.ArgumentParser(description='Simple parser')
    init(parser)
    main(args=parser.parse_args())
