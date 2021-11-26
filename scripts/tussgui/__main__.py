import argparse

from . import plot


def init(parser):
    subparsers = parser.add_subparsers(dest='target', required=True)

    parser_still = subparsers.add_parser('still')
    parser_still.add_argument('dir', type=str,
                              help='path to BODY_STATE_VECs')
    parser_still.add_argument('--max_iterations', default=-1, type=int,
                              help='max number of BODY_STATE_VECs to read')

    parser_live = subparsers.add_parser('live')
    parser_live.add_argument('dir', type=str,
                             help='path to BODY_STATE_VECs')
    parser_live.add_argument('--fps', default=200, type=int,
                             help='number of iterations to plot per second')
    # Add arg to control history of trajectory to keep


def main(args):
    if args.target == 'still':
        plot.plot_still_trajectory(args.dir, args.max_iterations)
    elif args.target == 'live':
        plot.plot_live_trajectory(args.dir, args.fps)


parser = argparse.ArgumentParser()
init(parser)
main(parser.parse_args())
