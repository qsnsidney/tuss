from os import path

DEFAULT_TRIALS = 1


def init_parser_parent(parser, default_engine_enum):
    parser.add_argument('-i', '--input', type=str, default=path.join(
        path.dirname(path.realpath(__file__)), '../../data/ic/s0_s112500_g100000_d100000.bin'))
    parser.add_argument('-o', '--output', type=str, default=path.join(
        path.dirname(path.realpath(__file__)), '../../tmp'))
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'number of runs for each configuration (default: {DEFAULT_TRIALS})')
    parser.add_argument('--version', type=int, default=default_engine_enum.value,
                        help=f'version of enginer to test. default = {default_engine_enum.value}({default_engine_enum.name})')
