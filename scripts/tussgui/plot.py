import itertools
import os

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .. import core
from . import data


def plot_snapshot(bin_file_path):
    # Create figure
    fig = plt.figure()
    fig.suptitle(bin_file_path)
    fig.set_size_inches(16/1.3, 9/1.3)

    ax1 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1)

    system_state = core.serde.deserialize_system_state_from_bin(bin_file_path)
    pos_xs, pos_ys, pos_zs, _, _, _, masses = zip(*system_state)
    log_masses = np.log10(masses)

    print('Info:', 'len(pos_xs):', len(pos_xs))
    print('Info:', 'len(pos_ys):', len(pos_ys))
    print('Info:', 'len(pos_zs):', len(pos_zs))
    print('Info:', 'len(masses):', len(masses))

    ax1.set_title('XY Plane')
    xy_scatter = ax1.scatter(pos_xs, pos_ys, c=log_masses, s=log_masses-np.min(log_masses),
                             marker='.', edgecolor='none')
    xy_cb = fig.colorbar(xy_scatter, ax=ax1)
    xy_cb.set_label('$\log_{10}$ Mass')

    ax2.set_title('XZ Plane')
    xz_scatter = ax2.scatter(pos_xs, pos_zs, c=log_masses, s=log_masses-np.min(log_masses),
                             marker='.', edgecolor='none')
    xz_cb = fig.colorbar(xz_scatter, ax=ax2)
    xz_cb.set_label('$\log_{10}$ Mass')

    ax3.set_title('YZ Plane')
    yz_scatter = ax3.scatter(pos_ys, pos_zs, c=log_masses, s=log_masses-np.min(log_masses),
                             marker='.', edgecolor='none')
    yz_cb = fig.colorbar(yz_scatter, ax=ax3)
    yz_cb.set_label('$\log_{10}$ Mass')

    plt.tight_layout()
    plt.show()


def plot_still_trajectory(dir, max_iterations=-1):
    # Create figure
    fig = plt.figure()

    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")

    system_state_batch = data.fetch_batch_system_state_all(
        dir, max_iterations)
    print('Info:', len(system_state_batch), 'Iterations')

    time_series_bodies = data.transform_batch_system_state_to_time_series(
        system_state_batch)
    n_bodies = len(time_series_bodies)
    print('Info:', n_bodies, "Bodies")

    # Plot the orbits
    for time_series_body in time_series_bodies:
        ax.plot(time_series_body[0],
                time_series_body[1], time_series_body[2])

    # Plot the final positions of the stars
    for body_i, time_series_body in enumerate(time_series_bodies):
        ax.scatter(time_series_body[0][-1], time_series_body[1][-1], time_series_body[2][-1],
                   marker='o', s=10, label='Body ' + str(body_i))

    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    ax.set_zlabel("z-coordinate")
    ax.set_title('Visualization of orbits of bodies in a ' +
                 str(n_bodies) + '-body system\n', fontsize=8)
    if n_bodies < 10:
        ax.legend()

    plt.show()


def plot_live_trajectory(dir, fps):
    # define sleep_handler
    def sleep_handler(x):
        if x > 0:
            plt.pause(x)

    # Create figure
    fig = plt.figure()

    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")

    # Init
    n_bodies = len(data.fetch_system_state(
        dir, 0, sleep_handler=sleep_handler))
    orbits = list()
    for body_i in range(n_bodies):
        line, = ax.plot([], [], [], label='Body ' + str(body_i))
        orbits.append(line)

    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    ax.set_zlabel("z-coordinate")
    ax.set_title('Visualization of orbits of bodies in a ' +
                 str(n_bodies) + '-body system\n', fontsize=8)
    if n_bodies < 10:
        ax.legend()

    # Axis limit
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    min_y = float('inf')
    max_z = float('-inf')
    min_z = float('inf')

    # frame_interval calculation
    frame_interval = 1000.0 / fps
    print('Info:', 'fps=' + str(fps),
          'intended_frame_interval=' + str(frame_interval))
    frame_multiplier = 1
    min_frame_interval = 5.0
    if frame_interval < min_frame_interval:
        frame_multiplier = min_frame_interval / frame_interval
        frame_interval = min_frame_interval
    print('Info:', 'frame_interval=' + str(frame_interval),
          'frame_multiplier=' + str(frame_multiplier))

    def init():
        nonlocal orbits
        for orbit in orbits:
            orbit.set_xdata(np.array([]))
            orbit.set_ydata(np.array([]))
            orbit.set_3d_properties(np.array([]))
        return orbits

    def animate(i_frame):
        nonlocal sleep_handler
        nonlocal frame_multiplier
        nonlocal ax
        nonlocal dir
        nonlocal orbits
        nonlocal max_x
        nonlocal min_x
        nonlocal max_y
        nonlocal min_y
        nonlocal max_z
        nonlocal min_z

        i_iter = int(i_frame * frame_multiplier)
        system_state = data.fetch_system_state(
            dir, i_iter, sleep_handler=sleep_handler)

        # Plot the orbits
        for body_i, body_state in enumerate(system_state):
            max_x = max(max_x, body_state[0]+1)
            min_x = min(min_x, body_state[0]-1)
            max_y = max(max_y, body_state[1]+1)
            min_y = min(min_y, body_state[1]-1)
            max_z = max(max_z, body_state[2]+1)
            min_z = min(min_z, body_state[2]-1)

            orbits[body_i].set_xdata(
                np.append(orbits[body_i].get_data_3d()[0], body_state[0]))
            orbits[body_i].set_ydata(
                np.append(orbits[body_i].get_data_3d()[1], body_state[1]))
            orbits[body_i].set_3d_properties(
                np.append(orbits[body_i].get_data_3d()[2], body_state[2]))

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        return orbits

    # TODO: fix the blit=True issue
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=itertools.count(), interval=frame_interval, blit=False, cache_frame_data=False)

    plt.show()
    # Can save as video format
