import argparse

from . import cuda


def init(parser):
    subparsers = parser.add_subparsers(dest='target', required=True)

    parser_cuda = subparsers.add_parser('cuda')
    cuda.init(parser_cuda)


def main(args):
    if args.target == 'cuda':
        cuda.main_new(args=args)


parser = argparse.ArgumentParser()
init(parser)
main(parser.parse_args())
