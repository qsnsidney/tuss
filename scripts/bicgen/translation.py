try:
    import yt
except Exception as e:
    print('Try `pip3 install yt`')
    print('Try `pip3 install pooch`')
    print('Try `pip3 install pandas`')

from .. import core

# https://yt-project.org/doc/cookbook/tipsy_notebook.html


def from_tipsy_into_bin(in_tipsy_file_path, out_bin_file_path, body_types=None):
    print('Info:', 'Loading from tipsy', in_tipsy_file_path)
    print('Info:', 'Writing into bin', out_bin_file_path)
    system_state = deserialize_system_state_from_tipsy(
        in_tipsy_file_path, body_types=body_types)
    core.serde.serialize_system_state(system_state, out_bin_file_path)


def deserialize_system_state_from_tipsy(tipsy_file_path=None, body_types=None):
    '''
    Download and use a sample from yt project if tipsy_file_path=None
    body_types: iteratable or None (for all body types)

    Original values are normalized to equivalent numerical values in following units, such that
    with the following values, the constant G for calculating gravity has a numerical value 1:
    mass: [in Msun] * G_solar_mass_parsec_kmps(4.3009e-3)
    distance: [in ps]
    velocity: [in km/s]

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

    if len(body_types) == 0:
        return list()

    ad = ds.all_data()
    system_state = list()
    for body_type in body_types:
        pos_xs = ad[body_type, 'Coordinates'][:, 0].to('pc').v
        pos_ys = ad[body_type, 'Coordinates'][:, 1].to('pc').v
        pos_zs = ad[body_type, 'Coordinates'][:, 2].to('pc').v

        vel_xs = ad[body_type, 'Velocities'][:, 0].to('km/s').v
        vel_ys = ad[body_type, 'Velocities'][:, 1].to('km/s').v
        vel_zs = ad[body_type, 'Velocities'][:, 2].to('km/s').v

        masses = ad[body_type, 'Mass'][:].to('Msun').v * 4.3009e-3

        prev_len_system_state = len(system_state)
        system_state.extend(zip(pos_xs, pos_ys, pos_zs,
                                vel_xs, vel_ys, vel_zs, masses))
        print('Info:', 'Num bodies loaded -', str(
            body_type) + ':', len(system_state)-prev_len_system_state)

    print('Info:', 'Total num bodies loaded:', len(system_state))
    return system_state
