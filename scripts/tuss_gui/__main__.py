from .. import core
import os
from matplotlib import pyplot as plt
import numpy as np


def read_body_state_vec(dir, i):
    body_state_vec_bin_file = os.path.join(dir, str(i) + '.bin')
    # print('Reading', body_state_vec_bin_file)
    if os.path.isfile(body_state_vec_bin_file):
        # print('Debug:', 'Successfully read', body_state_vec_bin_file)
        return core.serde.deserialize_body_state_vec_from_bin(
            body_state_vec_bin_file)
    else:
        return None


def fetch_batch_body_state_vec(dir, start_iteration, trajectory_iteration_batch_size):
    print('Info:', 'Fetching', trajectory_iteration_batch_size, 'from', dir)
    body_state_vec_batch = list()
    for i in range(start_iteration, start_iteration + trajectory_iteration_batch_size):
        body_state_vec_i = None
        # Retry fetching
        retry_count = 0
        while body_state_vec_i is None:
            body_state_vec_i = read_body_state_vec(dir, i)
            retry_count += 1
            assert retry_count < 10
        body_state_vec_batch.append(body_state_vec_i)
    print('Info:', '    Fetched')
    return body_state_vec_batch


def transform_batch_body_state_vec_to_time_series(body_state_vec_batch):
    '''
    [(pos_x_over_time, pos_y_over_time, pos_z_over_time) for each body]
    '''
    num_bodies = len(body_state_vec_batch[0])
    result = list()
    for body_i in range(num_bodies):
        pos_x_list = list()
        pos_y_list = list()
        pos_z_list = list()
        for body_state_vec in body_state_vec_batch:
            pos_x, pos_y, pos_z = body_state_vec[body_i][0:3]
            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)
            pos_z_list.append(pos_z)
        result.append((np.array(pos_x_list), np.array(
            pos_y_list), np.array(pos_z_list)))
    return result


def plot_live_trajectory(dir, trajectory_iteration_batch_size):
    # Create figure
    fig = plt.figure()

    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")

    # Initial
    iteration = 0
    time_series_bodies = transform_batch_body_state_vec_to_time_series(
        fetch_batch_body_state_vec(dir, iteration, trajectory_iteration_batch_size))
    n_bodies = len(time_series_bodies)

    # Plot the orbits
    orbits = list()
    for time_series_body in time_series_bodies:
        line, = ax.plot(time_series_body[0],
                        time_series_body[1], time_series_body[2])
        orbits.append(line)

    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    ax.set_zlabel("z-coordinate")
    ax.set_title('Visualization of orbits of bodies in a ' +
                 str(n_bodies) + ' system\n', fontsize=8)
    # ax.legend(loc="upper left", fontsize=14)

    plt.ion()
    plt.show()

    while True:
        plt.pause(0.5)

        iteration += trajectory_iteration_batch_size
        time_series_bodies = transform_batch_body_state_vec_to_time_series(
            fetch_batch_body_state_vec(dir, iteration, trajectory_iteration_batch_size))

        # Plot the orbits
        for body_i, time_series_body in enumerate(time_series_bodies):
            # orbits[body_i].set_xdata(time_series_body[0])
            # orbits[body_i].set_ydata(time_series_body[1])
            # orbits[body_i].set_3d_properties(time_series_body[2])
            # if body_i == 0:
            #     print(orbits[body_i].get_data_3d())
            #     if iteration == 10:
            #         assert False
            # if body_i == 0:
            #     # print('get_data_3d', orbits[body_i].get_data_3d())
            #     print('get_data_3d[0]', orbits[body_i].get_data_3d()[0])
            #     print('type(get_data_3d[0])', type(
            #         orbits[body_i].get_data_3d()[0]))
            #     # print('get_xdata', orbits[body_i].get_xdata())
            #     print('time_series_body[0]', time_series_body[0])
            #     print('type(time_series_body[0]0', type(time_series_body[0]))
            #     print('orbits[body_i].get_data_3d()[0] + time_series_body[0]',
            #           orbits[body_i].get_data_3d()[0] + time_series_body[0])
            #     if iteration == 3:
            #         assert False
            orbits[body_i].set_xdata(
                np.append(orbits[body_i].get_data_3d()[0], time_series_body[0]))
            orbits[body_i].set_ydata(
                np.append(orbits[body_i].get_data_3d()[1], time_series_body[1]))
            orbits[body_i].set_3d_properties(
                np.append(orbits[body_i].get_data_3d()[2], time_series_body[2]))
            # if body_i == 0:
            #     print('get_data_3d', orbits[body_i].get_data_3d())
            #     print('get_xdata', orbits[body_i].get_xdata())
            #     if iteration == 3:
            #         assert False

        ax.relim()
        # ax.autoscale()
        ax.autoscale()
        # ax.figure.canvas.draw_idle()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # # Plot the final positions of the stars
        # for body_i, time_series_body in enumerate(time_series_bodies):
        #     ax.scatter(time_series_body[0][-1], time_series_body[1][-1], time_series_body[2][-1],
        #                marker='o', s=10, label='Body ' + str(body_i))


plot_live_trajectory('./tmp/10_body_log', 1)
