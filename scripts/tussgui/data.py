import os

import numpy as np

from .. import core


def read_body_state_vec(dir, i):
    body_state_vec_bin_file = os.path.join(dir, str(i) + '.bin')
    # print('Reading', body_state_vec_bin_file)
    if os.path.isfile(body_state_vec_bin_file):
        # print('Debug:', 'Successfully read', body_state_vec_bin_file)
        return core.serde.deserialize_body_state_vec_from_bin(
            body_state_vec_bin_file)
    else:
        return None


def fetch_body_state_vec(dir, i, sleep_handler=None):
    body_state_vec_bin_file = os.path.join(dir, str(i) + '.bin')
    print('Info:', 'Fetching', body_state_vec_bin_file, 'from', dir)
    body_state_vec_i = None
    # Retry fetching
    retry_count = 0
    while body_state_vec_i is None:
        if sleep_handler is not None:
            sleep_handler(retry_count)
        body_state_vec_i = read_body_state_vec(dir, i)
        retry_count += 1
        # assert retry_count < 10
    return body_state_vec_i


def fetch_batch_body_state_vec(dir, start_iteration, trajectory_iteration_batch_size, no_retry=False, sleep_handler=None):
    print('Info:', 'Fetching', trajectory_iteration_batch_size,
          'BODY_STATE_VECs from', dir)
    body_state_vec_batch = list()
    for i in range(start_iteration, start_iteration + trajectory_iteration_batch_size):
        body_state_vec_i = None

        # Retry fetching
        retry_count = 0
        while body_state_vec_i is None:
            assert no_retry and retry_count == 0
            if sleep_handler is not None:
                sleep_handler(retry_count)
            body_state_vec_i = read_body_state_vec(dir, i)
            retry_count += 1
            # assert retry_count < 10
        body_state_vec_batch.append(body_state_vec_i)

    print('Info:', '    Fetched')
    return body_state_vec_batch


def fetch_batch_body_state_vec_all(dir, max_iterations=-1):
    num_files = len(core.fileio.files_in_dir(dir))
    print('Info:', 'Found', num_files, 'BIN files')
    if max_iterations >= 0:
        num_files = min(num_files, max_iterations)
    return fetch_batch_body_state_vec(dir, 0, num_files, no_retry=True)


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
