# /// BINARY
# /// - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
# /// - second 4 bytes: number of bodies
# /// - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_STATE
# /// Everything in binary

# void serialize_system_state_to_bin(std::ostream &, const SYSTEM_STATE &);
# void serialize_system_state_to_bin(const std::string &, const SYSTEM_STATE &);

# SYSTEM_STATE deserialize_system_state_from_bin(std::istream &);
# SYSTEM_STATE deserialize_system_state_from_bin(const std::string &);
import struct

from . import fileio


def parse_body_state_from_bin(f, floating_type_size, floating_type_sym):
    '''
    (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)
    '''
    body_state_bytes = f.read(floating_type_size * 7)
    return struct.unpack(floating_type_sym * 7, body_state_bytes)


def deserialize_system_state_from_bin(filename):
    '''
    [(POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)]
    '''
    with open(filename, 'rb') as f:
        floating_type_size = int.from_bytes(f.read(4), 'little')
        assert floating_type_size == 4 or floating_type_size == 8
        floating_type_sym = 'f' if floating_type_size == 4 else 'd'
        # print(floating_type_size, floating_type_sym)

        num_bodies = int.from_bytes(f.read(4), 'little')
        # print(num_bodies)

        return [parse_body_state_from_bin(f, floating_type_size, floating_type_sym) for _ in range(num_bodies)]


def write_body_state_into_bin(f, floating_type_sym, body_state):
    '''
    (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)
    '''
    f.write(struct.pack(floating_type_sym*7, *body_state))


def serialize_system_state_into_bin(system_state, filename):
    '''
    [(POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS)]
    '''
    fileio.create_file_dir_if_necessary(filename)
    with open(filename, 'wb') as f:
        floating_type_sym = 'f' if type(system_state[0][0]) is float else 'd'
        floating_type_size = 4 if floating_type_sym == 'f' else 8
        print('Info:', "BIN file floating type size", floating_type_size)
        f.write(floating_type_size.to_bytes(4, "little"))
        f.write(len(system_state).to_bytes(4, 'little'))
        for body_state in system_state:
            write_body_state_into_bin(f, floating_type_sym, body_state)
