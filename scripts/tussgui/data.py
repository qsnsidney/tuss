import os

import numpy as np

from .. import core


def read_system_state(dir, i):
    system_state_bin_file = os.path.join(dir, str(i) + '.bin')
    # print('Reading', system_state_bin_file)
    if os.path.isfile(system_state_bin_file):
        # print('Debug:', 'Successfully read', system_state_bin_file)
        return core.serde.deserialize_system_state_from_bin(
            system_state_bin_file)
    else:
        return None


def fetch_system_state(dir, i, sleep_handler=None):
    system_state_bin_file = os.path.join(dir, str(i) + '.bin')
    print('Info:', 'Fetching', system_state_bin_file, 'from', dir)
    system_state_i = None
    # Retry fetching
    retry_count = 0
    while system_state_i is None:
        if sleep_handler is not None:
            sleep_handler(retry_count)
        system_state_i = read_system_state(dir, i)
        retry_count += 1
        # assert retry_count < 10
    return system_state_i


def fetch_batch_system_state(dir, start_iteration, trajectory_iteration_batch_size, no_retry=False, sleep_handler=None):
    print('Info:', 'Fetching', trajectory_iteration_batch_size,
          'system_states from', dir)
    system_state_batch = list()
    for i in range(start_iteration, start_iteration + trajectory_iteration_batch_size):
        system_state_i = None

        # Retry fetching
        retry_count = 0
        while system_state_i is None:
            assert no_retry and retry_count == 0
            if sleep_handler is not None:
                sleep_handler(retry_count)
            system_state_i = read_system_state(dir, i)
            retry_count += 1
            # assert retry_count < 10
        system_state_batch.append(system_state_i)

    print('Info:', '    Fetched')
    return system_state_batch


def fetch_batch_system_state_all(dir, max_iterations=-1):
    num_files = len(core.fileio.files_in_dir(dir))
    print('Info:', 'Found', num_files, 'BIN files')
    if max_iterations >= 0:
        num_files = min(num_files, max_iterations)
    return fetch_batch_system_state(dir, 0, num_files, no_retry=True)


def transform_batch_system_state_to_time_series(system_state_batch):
    '''
    [(pos_x_over_time, pos_y_over_time, pos_z_over_time) for each body]
    '''
    num_bodies = len(system_state_batch[0])
    result = list()
    for body_i in range(num_bodies):
        pos_x_list = list()
        pos_y_list = list()
        pos_z_list = list()
        for system_state in system_state_batch:
            pos_x, pos_y, pos_z = system_state[body_i][0:3]
            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)
            pos_z_list.append(pos_z)
        result.append((np.array(pos_x_list), np.array(
            pos_y_list), np.array(pos_z_list)))
    return result
