from os import path

DEFAULT_TRIALS = 1


def get_script_path():
    return path.join(path.dirname(path.realpath(__file__)),
                     '../../')


def init_parser_parent(parser, default_exe, default_engine_enum):
    parser.add_argument('-i', '--input', type=str, default=path.join(
        get_script_path(), 'data/ic/s0_s112500_g100000_d100000.bin'))
    parser.add_argument('-o', '--output', type=str, default=path.join(
        get_script_path(), 'tmp'))
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'number of runs for each configuration (default: {DEFAULT_TRIALS})')
    parser.add_argument('--version', type=int, default=default_engine_enum.value,
                        help=f'version of engine to test. default = {default_engine_enum.value}({default_engine_enum.name})')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--exe', type=str, default=default_exe,
                        help=f'engine executable. default = {default_exe}')
    parser.add_argument('-m', '--message', type=str, default=None,
                        help=f'Labels/messages for this experiment run')
