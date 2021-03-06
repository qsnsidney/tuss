import re
import subprocess
from os import path
from typing import NamedTuple


class SchedulerParams(NamedTuple):
    suite_name: str
    engine_version: int
    engine_name: str
    executable: str
    exe_args_sweep: dict  # {arg_name: [arg]}
    num_trials_per_run: int
    wdir: str
    timer_pattern: str
    benchmark_data_path: str
    dt: float
    num_iterations: int


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
    '''
    suite_result = [(exe_arg_line, avg_time)]
    '''
    suite_result = list()
    STDOUT_OUTPUT = 'benchmark.stdout'

    print(
        f'Running {args.suite_name} benchmark for {args.engine_name} ({args.engine_version})')

    exe_args_sweep = permutate_exe_args(
        validate_exe_args(args.exe_args_sweep))

    stdout_file_path = path.join(args.wdir, STDOUT_OUTPUT)
    print(f'Creating file for stdout {stdout_file_path}')
    with open(stdout_file_path, 'w') as f_stdout:
        for exe_arg_line in exe_args_sweep:
            total_time = 0
            assert args.num_trials_per_run > 0
            for itrial in range(args.num_trials_per_run):
                command = [args.executable] + exe_arg_line + [
                    '--ic_file', args.benchmark_data_path,
                    '--dt', str(args.dt),
                    '--num_iterations', str(args.num_iterations),
                    '--version', str(args.engine_version)]

                command_str = ' '.join(command)
                info_msg = f'Running {command_str}, trial {itrial}'
                f_stdout.write(info_msg + '\n')
                print(info_msg)

                result = None
                try:
                    result = subprocess.check_output(
                        command, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    ret = e.output.decode('utf-8')
                    print('---------------------------------------------------------')
                    print('[Error Running]')
                    print(' '.join(command))
                    print(' ')
                    print('[Error]')
                    print(ret)
                    print('---------------------------------------------------------')
                    time_elapsed = None
                else:
                    ret = result.decode('utf-8')
                    timer_match = re.search(args.timer_pattern, ret)
                    if not timer_match:
                        raise(Exception('failed to find timer dump'))
                    time_elapsed = float(timer_match.group(1))

                info_msg = f'    {time_elapsed}'
                f_stdout.write(info_msg + '\n')
                print(info_msg)
                f_stdout.write(ret + '\n')
                f_stdout.write(
                    '=========================================================\n\n')

                if time_elapsed is not None and total_time is not None:
                    total_time += time_elapsed
                else:
                    total_time = None

                f_stdout.flush()

            if total_time is not None:
                avg_time = total_time / args.num_trials_per_run
            else:
                avg_time = None
            suite_result.append((command, avg_time))

    return suite_result


def write_suite_result_to_file(suite_result, data_output_file_path, num_iterations, message=None):
    print(f'Creating file for result {data_output_file_path}')
    print('RESULT')
    with open(data_output_file_path, 'w') as f_data:
        f_data.write('Message:\n')
        if message is not None:
            f_data.write(message)
        else:
            f_data.write('None')
        f_data.write('\n\n')

        for exe_arg_line, avg_time in suite_result:
            iter_time = (
                avg_time / num_iterations) if avg_time is not None else None
            line = str(exe_arg_line) + ': '
            line += '{:.6f}'.format(avg_time) if avg_time is not None else 'None'
            line += ', ' + ('{:.6f}'.format(iter_time)
                            if iter_time is not None else 'None')
            f_data.write(line + '\n')
            print(line)
