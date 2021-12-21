import argparse

from . import cuda, scheduler


def init(parser):
    subparsers = parser.add_subparsers(dest='target', required=True)

    parser_cuda = subparsers.add_parser('cuda')
    cuda.init(parser_cuda)


def main(args):
    if args.target == 'cuda':
        scheduler.main(args=args)


parser = argparse.ArgumentParser()
init(parser)
main(parser.parse_args())
