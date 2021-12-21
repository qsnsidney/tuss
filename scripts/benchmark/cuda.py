#!/usr/bin/env python3

import argparse
import re
import subprocess
from enum import Enum
from os import path

from .. import core
from . import scheduler


class CudaEngine(Enum):
    BASIC = 0
    NVDA_REFERENCE = 1
    COALESCED_BASIC = 2
    TILED_BASIC = 3
    SMALL_N = 4
    MAT_MUL = 5
    NVDA_IMPROVED = 6


DEFAULT_TRIALS = 1


def init(parser):
    parser.add_argument('-i', '--input', type=str, default=path.join(
        path.dirname(path.realpath(__file__)), '../../data/ic/s0_s112500_g100000_d100000.bin'))
    parser.add_argument('-o', '--output', type=str, default=path.join(
        path.dirname(path.realpath(__file__)), '../../tmp'))
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'number of runs for each configuration (default: {DEFAULT_TRIALS})')
    parser.add_argument('--version', type=int, default=CudaEngine.NVDA_IMPROVED.value,
                        help=f'version of enginer to test. default = {CudaEngine.NVDA_IMPROVED.value}({CudaEngine.NVDA_IMPROVED.name})')


def main(args):
    script_path = path.join(path.dirname(path.realpath(__file__)),
                            '../../')
    project_home_dir = path.join(
        args.output, core.utils.get_pf_timestamp_str())
    core.fileio.create_dir_if_necessary(project_home_dir)

    print(f'Using {project_home_dir} as working directory')

    scheduler_args = scheduler.SchedulerParams(
        'GPU',
        args.version,
        CudaEngine(args.version).name,
        path.join(script_path, 'build/tus/tus_exe'),
        {'--num_bodies': [
            50000, 100000], '--block_size': [16, 64, 256]},
        args.trials,
        project_home_dir,
        'Profile \[all_iters\]: (([0-9]*[.])?[0-9]+)',
        args.input)

    result = scheduler.schedule_run(scheduler_args)

    BENCHMARK_OUTPUT_FILE = f'{scheduler_args.suite_name.lower()}_benchmark_{CudaEngine(scheduler_args.engine_version).name}.csv'
    data_output_file_path = path.join(project_home_dir, BENCHMARK_OUTPUT_FILE)
    print(' ')
    print('RESULT')
    with open(data_output_file_path, 'w') as f_data:
        for exe_arg_line, avg_time in result:
            line = str(exe_arg_line) + ': ' + '{:.6f}'.format(avg_time)
            f_data.write(line + '\n')
            print(line)


def main_deprecated(args):
    '''
    @Deprecated
    '''
    THREAD_PER_BLOCK = [16, 64, 256]
    NBODY = [50000, 100000]
    CUDA_EXECUTABLE = 'build/tus/tus_exe'
    GPU_TIME_PATTERN = 'Profile \[all_iters\]: (([0-9]*[.])?[0-9]+)'
    BENCHMARK_DATA = 'data/ic/s0_s112500_g100000_d100000.bin'
    STDOUT_OUTPUT = 'benchmark.stdout'
    VERSION = args.version
    AVG_ITERATION = args.trials
    BENCHMARK_OUTPUT_FILE = f'gpu_benchmark_{CudaEngine(VERSION).name}.csv'

    print(f'running benchmark for {CudaEngine(VERSION).name}')

    script_dir = path.dirname(path.realpath(__file__))
    project_home_dir = path.join(script_dir, '../../')

    data_output_file_path = path.join(
        project_home_dir, BENCHMARK_OUTPUT_FILE)
    stdout_file_path = path.join(project_home_dir, STDOUT_OUTPUT)
    benchmark_path = path.join(project_home_dir, BENCHMARK_DATA)
    cuda_executable = path.join(project_home_dir, CUDA_EXECUTABLE)

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
                    raise(Exception(e.output.decode('utf-8')))

                ret = result.decode('utf-8')
                f_stdout.write(ret + '\n')
                gpu_runtime_re = re.search(GPU_TIME_PATTERN, ret)
                if not gpu_runtime_re:
                    raise(Exception('failed to find gpu runtime'))
                gpu_runtime = gpu_runtime_re.group(1)
                total_time += float(gpu_runtime)
            avg_time = total_time / AVG_ITERATION
            f_data.write('{:.6f}'.format(avg_time) + ',')

    f_data.close()
    f_stdout.close()


if __name__ == '__main__':
    '''
    @Deprecated
    '''
    print('WARNING:', 'Launching directly from this script is now deprecated!')
    print('WARNING:', 'Please use:')
    print('WARNING:', '  python3 -m scripts.benchmark cuda')
    print(' ')
    parser = argparse.ArgumentParser(description='Simple parser')
    init(parser)
    main_deprecated(args=parser.parse_args())
