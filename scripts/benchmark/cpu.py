from enum import Enum
from os import path

from .. import core
from . import general, scheduler


class CpuEngine(Enum):
    BASIC = 0
    SHARED_ACC = 1


DEFAULT_TRIALS = 1


def init(parser):
    general.init_parser_parent(parser, CpuEngine.SHARED_ACC)


def main(args):
    script_path = path.join(path.dirname(path.realpath(__file__)),
                            '../../')
    project_home_dir = path.join(
        args.output, core.utils.get_pf_timestamp_str())
    core.fileio.create_dir_if_necessary(project_home_dir)

    print(f'Using {project_home_dir} as working directory')

    scheduler_args = scheduler.SchedulerParams(
        'CPU',
        args.version,
        CpuEngine(args.version).name,
        path.join(script_path, 'build/cpusim/cpusim_exe'),
        {'--num_bodies': [5000, 10000],
         '--num_threads': [1, 2, 4]},
        args.trials,
        project_home_dir,
        'Subprofile \[.*/all_iters\]: (([0-9]*[.])?[0-9]+)',
        args.input)

    result = scheduler.schedule_run(scheduler_args)

    BENCHMARK_OUTPUT_FILE = f'{scheduler_args.suite_name.lower()}_benchmark_{CpuEngine(scheduler_args.engine_version).value}.csv'
    data_output_file_path = path.join(project_home_dir, BENCHMARK_OUTPUT_FILE)
    print(' ')
    print('RESULT')
    with open(data_output_file_path, 'w') as f_data:
        for exe_arg_line, avg_time in result:
            line = str(exe_arg_line) + ': ' + \
                ('{:.6f}'.format(avg_time) if avg_time is not None else 'None')
            f_data.write(line + '\n')
            print(line)
