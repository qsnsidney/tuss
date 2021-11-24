import argparse

from . import plot


def init(parser):
    subparsers = parser.add_subparsers(dest='target', required=True)

    parser_fixed = subparsers.add_parser('fixed')
    parser_fixed.add_argument('dir', type=str,
                              help='path to BODY_STATE_VECs')
    parser_fixed.add_argument('--max_iterations', default=-1,
                              help='max number of BODY_STATE_VECs to read')

    parser_live = subparsers.add_parser('live')
    parser_live.add_argument('dir', type=str,
                             help='path to BODY_STATE_VECs')
    parser_live.add_argument('--fps', default=100,
                             help='number of iterations to plot per second')


def main(args):
    if args.target == 'fixed':
        plot.plot_fixed_trajectory(args.dir, args.max_iterations)
    elif args.target == 'live':
        plot.plot_live_trajectory(args.dir, args.fps)


parser = argparse.ArgumentParser()
init(parser)
main(parser.parse_args())
