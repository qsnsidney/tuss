from enum import Enum
from os import path

from .. import core
from . import scheduler


class CpuEngine(Enum):
    BASIC = 0
    SHARED_ACC = 1


DEFAULT_TRIALS = 1


def init(parser):
    parser.add_argument('-i', '--input', type=str, default=path.join(
        path.dirname(path.realpath(__file__)), '../../data/ic/s0_s112500_g100000_d100000.bin'))
    parser.add_argument('-o', '--output', type=str, default=path.join(
        path.dirname(path.realpath(__file__)), '../../tmp'))
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'number of runs for each configuration (default: {DEFAULT_TRIALS})')
    parser.add_argument('--version', type=int, default=CpuEngine.SHARED_ACC.value,
                        help=f'version of enginer to test. default = {CpuEngine.SHARED_ACC.value}({CpuEngine.SHARED_ACC.name})')


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
        {'--num_bodies': [
            5000, 10000], '--num_threads': [1, 2, 4]},
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
            line = str(exe_arg_line) + ': ' + '{:.6f}'.format(avg_time)
            f_data.write(line + '\n')
            print(line)
