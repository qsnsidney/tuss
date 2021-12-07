import argparse

from . import plot


def init(parser):
    subparsers = parser.add_subparsers(dest='target', required=True)

    parser_snapshot = subparsers.add_parser('snapshot')
    parser_snapshot.add_argument('bin_file', type=str,
                                 help='path to system_state bin file')

    parser_trajectory_still = subparsers.add_parser('trajectory_still')
    parser_trajectory_still.add_argument('dir', type=str,
                                         help='path to system_states')
    parser_trajectory_still.add_argument('--max_iterations', default=-1, type=int,
                                         help='max number of system_states to read')

    parser_trajectory_live = subparsers.add_parser('trajectory_live')
    parser_trajectory_live.add_argument('dir', type=str,
                                        help='path to system_states')
    parser_trajectory_live.add_argument('--fps', default=200, type=int,
                                        help='number of iterations to plot per second')
    # Add arg to control history of trajectory to keep


def main(args):
    if args.target == 'snapshot':
        plot.plot_snapshot(args.bin_file)
    elif args.target == 'trajectory_still':
        plot.plot_still_trajectory(args.dir, args.max_iterations)
    elif args.target == 'trajectory_live':
        plot.plot_live_trajectory(args.dir, args.fps)


parser = argparse.ArgumentParser()
init(parser)
main(parser.parse_args())
