import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .. import core
from . import data


def plot_fixed_trajectory(dir, max_iterations=-1):
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
    # Create figure
    fig = plt.figure()

    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")

    def sleep_handler(x):
        if x > 0:
            plt.pause(x)

    # Initial
    iteration = 0
    body_state_vec = data.fetch_body_state_vec(
        dir, iteration, sleep_handler=sleep_handler)
    n_bodies = len(body_state_vec)

    # Axis limit
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    max_z = 0
    min_z = 0

    # Plot the orbits
    orbits = list()
    for body_state in body_state_vec:
        max_x = max(max_x, body_state[0])
        min_x = min(min_x, body_state[0])
        max_y = max(max_y, body_state[1])
        min_y = min(min_y, body_state[1])
        max_z = max(max_z, body_state[2])
        min_z = min(min_z, body_state[2])

        line, = ax.plot(body_state[0],
                        body_state[1], body_state[2])
        orbits.append(line)

    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    ax.set_zlabel("z-coordinate")
    ax.set_title('Visualization of orbits of bodies in a ' +
                 str(n_bodies) + '-body system\n', fontsize=8)
    if n_bodies < 10:
        ax.legend()

    plt.ion()
    plt.show()

    while True:
        plt.pause(1.0/fps)

        iteration += 1
        body_state_vec = data.fetch_body_state_vec(
            dir, iteration, sleep_handler=sleep_handler)

        # Plot the orbits
        for body_i, body_state in enumerate(body_state_vec):
            max_x = max(max_x, body_state[0])
            min_x = min(min_x, body_state[0])
            max_y = max(max_y, body_state[1])
            min_y = min(min_y, body_state[1])
            max_z = max(max_z, body_state[2])
            min_z = min(min_z, body_state[2])

            orbits[body_i].set_xdata(
                np.append(orbits[body_i].get_data_3d()[0], body_state[0]))
            orbits[body_i].set_ydata(
                np.append(orbits[body_i].get_data_3d()[1], body_state[1]))
            orbits[body_i].set_3d_properties(
                np.append(orbits[body_i].get_data_3d()[2], body_state[2]))

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)

        # Doesn't work
        # fig.gca().relim()
        # fig.gca().autoscale()
        # fig.gca().autoscale_view(True,True,True)
        # ax.figure.canvas.draw_idle()

        fig.canvas.draw()
        fig.canvas.flush_events()
