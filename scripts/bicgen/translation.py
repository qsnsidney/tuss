try:
    import yt
except Exception as e:
    print('Try `pip3 install yt`')
    print('Try `pip3 install pooch`')
    print('Try `pip3 install pandas`')

from .. import core

# https://yt-project.org/doc/cookbook/tipsy_notebook.html


def from_tipsy_into_bin(in_tipsy_file_path, out_bin_file_path, body_types=None):
    system_state = deserialize_system_state_from_tipsy(
        in_tipsy_file_path, body_types=body_types)
    core.serde.serialize_system_state_into_bin(system_state, out_bin_file_path)


def deserialize_system_state_from_tipsy(tipsy_file_path=None, body_types=None):
    '''
    Download and use a sample from yt project if tipsy_file_path=None
    body_types: iteratable or None (for all body types)

    Return: [(POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)]
    '''
    if tipsy_file_path is None:
        ds = yt.load_sample('TipsyGalaxy')
    else:
        ds = yt.load(tipsy_file_path)

    # print(ds.field_list)

    if body_types is None:
        body_types = sorted(ds.particle_types_raw)
    else:
        body_types = sorted(
            set(body_types).intersection(ds.particle_types_raw))

    print('Info:', 'Available:', ds.particle_type_counts)
    print('Info:', 'Selected:', body_types)

    ad = ds.all_data()
    system_state = list()
    for body_type in body_types:
        pos_x_coords = ad[body_type, 'Coordinates'][:, 0].v
        pos_y_coords = ad[body_type, 'Coordinates'][:, 1].v
        pos_z_coords = ad[body_type, 'Coordinates'][:, 2].v

        vel_x_coords = ad[body_type, 'Velocities'][:, 0].v
        vel_y_coords = ad[body_type, 'Velocities'][:, 1].v
        vel_z_coords = ad[body_type, 'Velocities'][:, 2].v

        masses = ad[body_type, 'Mass'][:].v

        prev_len_system_state = len(system_state)
        system_state.extend(zip(pos_x_coords, pos_y_coords, pos_z_coords,
                                vel_x_coords, vel_y_coords, vel_z_coords, masses))
        print('Info:', 'Num bodies loaded -', str(
            body_type) + ':', len(system_state)-prev_len_system_state)

    print('Info:', 'Total num bodies loaded:', len(system_state))
    return system_state
