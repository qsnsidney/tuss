from collections import defaultdict
import os
from os import path
from typing import NamedTuple
import subprocess
import re

from scripts.benchmark.cuda import TEST_ENGINES_NAME

from .. import core


def error_and_exit(err_msg):
    print(err_msg)
    exit(1)


class SchedulerParams(NamedTuple):
    suite_name: str
    engine_version: int
    executable: str
    exe_args_sweep: dict  # {arg_name: [arg]}
    num_trials_per_run: int
    wdir: str
    timer_pattern: str
    benchmark_data_path: str


def main(args):
    script_path = path.join(path.dirname(path.realpath(__file__)),
                            '../../')
    scheduler_args = SchedulerParams(
        'GPU',
        args.version,
        path.join(script_path, 'build/tus/tus_exe'),
        {'--num_bodies': [
            50000, 100000], '--block_size': [16, 64, 256]},
        args.iter,
        args.output,
        'Profile \[all_iters\]: (([0-9]*[.])?[0-9]+)',
        path.join(script_path, 'data/ic/s0_s112500_g100000_d100000.bin'))

    result = schedule_run(scheduler_args)
    for row in result:
        print(row)
    # # THREAD_PER_BLOCK = [16, 64, 256]
    # # NBODY = [50000, 100000]
    # # CUDA_EXECUTABLE = 'build/tus/tus_exe'
    # # GPU_TIME_PATTERN = 'Profile \[all_iters\]: (([0-9]*[.])?[0-9]+)'
    # # BENCHMARK_DATA = 'data/ic/s0_s112500_g100000_d100000.bin'
    # STDOUT_OUTPUT = 'benchmark.stdout'
    # VERSION = args.version
    # # AVG_ITERATION = args.iter
    # BENCHMARK_OUTPUT_FILE = f'{scheduler_args.suite_name.lower()}_benchmark_{TEST_ENGINES_NAME[VERSION]}.csv'

    # print(f'running benchmark for {TEST_ENGINES_NAME[VERSION]}')

    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # project_home_dir = os.path.join(script_dir, '../../')

    # data_output_file_path = os.path.join(
    #     project_home_dir, BENCHMARK_OUTPUT_FILE)
    # stdout_file_path = os.path.join(project_home_dir, STDOUT_OUTPUT)
    # benchmark_path = os.path.join(project_home_dir, BENCHMARK_DATA)
    # cuda_executable = os.path.join(project_home_dir, CUDA_EXECUTABLE)

    # f_data = open(data_output_file_path, 'w')
    # f_stdout = open(stdout_file_path, 'w')

    # f_data.write('nbody/block')
    # for i in NBODY:
    #     f_data.write(',' + str(i))

    # for block_size in THREAD_PER_BLOCK:
    #     f_data.write('\n')
    #     f_data.write(str(block_size) + ',')
    #     for num_body in NBODY:
    #         total_time = 0
    #         for count in range(AVG_ITERATION):
    #             info_msg = 'RUNNING NUMBLOCK : ' + \
    #                 str(block_size) + '. NBODY : ' + \
    #                 str(num_body) + '. ITER: ' + str(count)
    #             f_stdout.write(info_msg + '\n')
    #             print(info_msg)
    #             command = [cuda_executable, '-b', str(num_body), '-i', benchmark_path, '-t', str(
    #                 block_size), '-d', '1', '-n', '10', '--version', str(VERSION)]
    #             try:
    #                 result = subprocess.check_output(
    #                     command, stderr=subprocess.STDOUT)
    #             except subprocess.CalledProcessError as e:
    #                 print(command)
    #                 error_and_exit(e.output.decode('utf-8'))

    #             ret = result.decode('utf-8')
    #             f_stdout.write(ret + '\n')
    #             gpu_runtime_re = re.search(GPU_TIME_PATTERN, ret)
    #             if not gpu_runtime_re:
    #                 error_and_exit('failed to find gpu runtime')
    #             gpu_runtime = gpu_runtime_re.group(1)
    #             total_time += float(gpu_runtime)
    #         avg_time = total_time / AVG_ITERATION
    #         f_data.write('{:.6f}'.format(avg_time) + ',')

    # f_data.close()
    # f_stdout.close()


def validate_exe_args(original_exe_args_sweep: dict):
    exe_arg_names_to_del = [
        k for k, v in original_exe_args_sweep.items() if len(v) == 0]
    new_exe_args_sweep = dict(original_exe_args_sweep)
    for arg_name in exe_arg_names_to_del:
        new_exe_args_sweep.pop(arg_name)
    return new_exe_args_sweep


def permutate_exe_args(exe_args_sweep: dict, exe_arg_names: list = None):
    '''
    {arg_name: [arg_value]}
    every arg_name must have non-empty arg_values
    '''
    if exe_arg_names is None:
        exe_arg_names = list(sorted(exe_args_sweep))

    l = list()
    if len(exe_arg_names) == 0:
        return l

    cur_exe_arg_name = exe_arg_names[0]
    for cur_exe_arg_value in exe_args_sweep[cur_exe_arg_name]:
        cur_arg_name_value = [cur_exe_arg_name, str(cur_exe_arg_value)]

        if len(exe_arg_names) == 1:
            l.append(cur_arg_name_value)
        else:
            post_arg_name_value_pairs = permutate_exe_args(
                exe_args_sweep, exe_arg_names[1:])
            for post_arg_name_value_pair in post_arg_name_value_pairs:
                l.append(cur_arg_name_value + post_arg_name_value_pair)
    return l


def schedule_run(args: SchedulerParams):
    suite_result = list()
    # BENCHMARK_OUTPUT_FILE = f'{args.suite_name.lower()}_benchmark_{args.engine_version}.csv'
    STDOUT_OUTPUT = 'benchmark.stdout'

    print(f'Running {args.suite_name} benchmark for {args.engine_version}')

    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # project_home_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
    project_home_dir = os.path.join(
        args.wdir, core.utils.get_pf_timestamp_str())
    core.fileio.create_dir_if_necessary(project_home_dir)

    # data_output_file_path = path.join(project_home_dir, BENCHMARK_OUTPUT_FILE)
    stdout_file_path = os.path.join(project_home_dir, STDOUT_OUTPUT)

    # with open(data_output_file_path, 'w') as f_data:
    exe_args_sweep = permutate_exe_args(
        validate_exe_args(args.exe_args_sweep))
    with open(stdout_file_path, 'w') as f_stdout:
        for exe_arg_line in exe_args_sweep:
            total_time = 0
            assert args.num_trials_per_run > 0
            for itrial in range(args.num_trials_per_run):
                info_msg = f'Running {exe_arg_line}, trial {itrial}'
                f_stdout.write(info_msg + '\n')
                print(info_msg)

                command = [args.executable] + exe_arg_line + ['--ic_file', args.benchmark_data_path, '--dt', str(
                    1), '--num_iterations', str(10), '--version', str(args.engine_version)]

                try:
                    result = subprocess.check_output(
                        command, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    print(command)
                    error_and_exit(e.output.decode('utf-8'))

                ret = result.decode('utf-8')
                timer_match = re.search(args.timer_pattern, ret)
                if not timer_match:
                    error_and_exit('failed to find gpu runtime')
                time_elapsed = float(timer_match.group(1))

                info_msg = f'    {time_elapsed}'
                f_stdout.write(info_msg + '\n')
                print(info_msg)
                f_stdout.write(ret + '\n')

                total_time += time_elapsed
            avg_time = total_time / args.num_trials_per_run
            suite_result.append((exe_arg_line, avg_time))

    return suite_result
