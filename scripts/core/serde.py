# /// BINARY
# /// - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
# /// - second 4 bytes: number of bodies
# /// - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_STATE
# /// Everything in binary

# void serialize_body_state_vec_to_bin(std::ostream &, const BODY_STATE_VEC &);
# void serialize_body_state_vec_to_bin(const std::string &, const BODY_STATE_VEC &);

# BODY_STATE_VEC deserialize_body_state_vec_from_bin(std::istream &);
# BODY_STATE_VEC deserialize_body_state_vec_from_bin(const std::string &);

def deserialize_body_state_vec_from_bin(filename):
    with open(filename, 'rb') as f:
        floating_type_size = int.from_bytes(f.read(4), 'little')
        assert floating_type_size == 4 or floating_type_size == 8
        print(floating_type_size)

        num_bodies = int.from_bytes(f.read(4), 'little')
        print(num_bodies)
