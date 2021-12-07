import argparse

from numpy import number
from . import translation

# IC Data
# https://nbody.shop/data.html


def init(parser):
    subparsers = parser.add_subparsers(dest='target', required=True)

    parser_translate = subparsers.add_parser('translate')
    parser_translate.add_argument('bin', type=str,
                                  help='path to bin output file')
    parser_translate.add_argument('--tipsy', default=None, type=str,
                                  help='path to tipsy input file. Optional')
    parser_translate.add_argument(
        '--body_types', type=str, nargs='*', default=None, help='body types of interest')


def main(args):
    if args.target == 'translate':
        print('Info:')
        print('Info:', 'args.tipsy', 'args.tipsy')
        print('Info:', 'args.bin', args.bin)
        print('Info:', 'args.body_types', args.body_types)
        print('Info:')
        translation.from_tipsy_into_bin(
            in_tipsy_file_path=args.tipsy, out_bin_file_path=args.bin, body_types=args.body_types)


parser = argparse.ArgumentParser()
init(parser)
main(parser.parse_args())
