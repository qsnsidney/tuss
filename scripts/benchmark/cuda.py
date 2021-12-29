#!/usr/bin/env python3

import argparse
import re
import subprocess
from enum import Enum
from os import path

from .. import core
from . import general, scheduler


class CudaEngine(Enum):
    BASIC = 0
    NVDA_REFERENCE = 1
    COALESCED_BASIC = 2
    TILED_BASIC = 3
    TILING_2D = 4
    MAT_MUL = 5
    NVDA_IMPROVED = 6


def init(parser):
    general.init_parser_parent(parser,  path.join(
        general.get_script_path(), 'build/tus/tus_exe'), CudaEngine.NVDA_IMPROVED)


def main(args):
    project_home_dir = path.join(
        args.output, core.utils.get_pf_timestamp_str())
    core.fileio.create_dir_if_necessary(project_home_dir)

    print(f'Using {project_home_dir} as working directory')

    exe_args_sweep = determine_exe_args_sweep(CudaEngine(args.version))

    scheduler_args = scheduler.SchedulerParams(
        'GPU',
        args.version,
        CudaEngine(args.version).name,
        args.exe,
        exe_args_sweep,
        args.trials,
        project_home_dir,
        'Profile \[all_iters\]: (([0-9]*[.])?[0-9]+)',
        args.input,
        args.dt,
        args.iterations)

    result = scheduler.schedule_run(scheduler_args)

    BENCHMARK_OUTPUT_FILE = f'{scheduler_args.suite_name.lower()}_benchmark_{CudaEngine(scheduler_args.engine_version).name}.csv'
    data_output_file_path = path.join(project_home_dir, BENCHMARK_OUTPUT_FILE)
    print(' ')
    scheduler.write_suite_result_to_file(
        result, data_output_file_path, args.iterations, args.message)


def determine_exe_args_sweep(version: CudaEngine):
    if version == CudaEngine.BASIC or version == CudaEngine.NVDA_REFERENCE:
        return {'--num_bodies': [20_000, 100_000, 200_000], '--block_size': [16, 64, 256]}
    elif version == CudaEngine.TILING_2D:
        return {'--num_bodies': [20_000, 100_000, 200_000],
                '--len': [1, 2, 4],
                '--wid': [64, 128, 256, 512],
                '--lur': [256, 512, 1024]}
    else:
        raise Exception('Even god cannot help you on this, sorry')


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

    for num_body in NBODY:
        f_data.write('\n')
        f_data.write(str(block_size) + ',')
        for block_size in THREAD_PER_BLOCK:
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
