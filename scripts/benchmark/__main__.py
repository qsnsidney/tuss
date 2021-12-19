import argparse

from . import cuda


def init(parser):
    subparsers = parser.add_subparsers(dest='target', required=True)

    parser_cuda = subparsers.add_parser('cuda')
    cuda.init_argparser(parser_cuda)


def main(args):
    if args.target == 'cuda':
        cuda.main(args=args)


parser = argparse.ArgumentParser()
init(parser)
main(parser.parse_args())
