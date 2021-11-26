import itertools
import os

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .. import core
from . import data


def plot_still_trajectory(dir, max_iterations=-1):
    # Create figure
    fig = plt.figure()

    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")

    body_state_vec_batch = data.fetch_batch_body_state_vec_all(
        dir, max_iterations)
    print('Info:', len(body_state_vec_batch), 'Iterations')

    time_series_bodies = data.transform_batch_body_state_vec_to_time_series(
        body_state_vec_batch)
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
    n_bodies = len(data.fetch_body_state_vec(
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
        body_state_vec = data.fetch_body_state_vec(
            dir, i_iter, sleep_handler=sleep_handler)

        # Plot the orbits
        for body_i, body_state in enumerate(body_state_vec):
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

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=itertools.count(), interval=frame_interval, blit=False, cache_frame_data=False)

    plt.show()
