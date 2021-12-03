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


def parse_body_state_from_bin(f, floating_type_size, floating_type_sym):
    body_state_bytes = f.read(floating_type_size * 7)
    return struct.unpack(floating_type_sym * 7, body_state_bytes)


def deserialize_system_state_from_bin(filename):
    with open(filename, 'rb') as f:
        floating_type_size = int.from_bytes(f.read(4), 'little')
        assert floating_type_size == 4 or floating_type_size == 8
        floating_type_sym = 'f' if floating_type_size == 4 else 'd'
        # print(floating_type_size, floating_type_sym)

        num_bodies = int.from_bytes(f.read(4), 'little')
        # print(num_bodies)

        return [parse_body_state_from_bin(f, floating_type_size, floating_type_sym) for _ in range(num_bodies)]
