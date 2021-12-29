from enum import Enum
from os import path

from .. import core
from . import general, scheduler


class CpuEngine(Enum):
    BASIC = 0
    SHARED_ACC = 1


DEFAULT_TRIALS = 1


def init(parser):
    general.init_parser_parent(parser, path.join(
        general.get_script_path(), 'build/cpusim/cpusim_exe'), CpuEngine.SHARED_ACC)


def main(args):
    project_home_dir = path.join(
        args.output, core.utils.get_pf_timestamp_str())
    core.fileio.create_dir_if_necessary(project_home_dir)

    print(f'Using {project_home_dir} as working directory')

    scheduler_args = scheduler.SchedulerParams(
        'CPU',
        args.version,
        CpuEngine(args.version).name,
        args.exe,
        {'--num_bodies': [100_000, 200_000], #[20_000, 100_000, 200_000],
         '--num_threads': [1, 4]},
        args.trials,
        project_home_dir,
        'Subprofile \[.*/all_iters\]: (([0-9]*[.])?[0-9]+)',
        args.input,
        args.dt,
        args.iterations)

    result = scheduler.schedule_run(scheduler_args)

    BENCHMARK_OUTPUT_FILE = f'{scheduler_args.suite_name.lower()}_benchmark_{CpuEngine(scheduler_args.engine_version).value}.csv'
    data_output_file_path = path.join(project_home_dir, BENCHMARK_OUTPUT_FILE)
    print(' ')
    scheduler.write_suite_result_to_file(
        result, data_output_file_path, args.iterations, args.message)
