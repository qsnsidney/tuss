#pragma once

#include <iostream>

#include "physics.h"

namespace CORE
{
    /// CSV
    /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row

    void serialize_body_ic_vec_to_csv(std::ostream &, const BODY_IC_VEC &);
    void serialize_body_ic_vec_to_csv(const std::string &, const BODY_IC_VEC &);

    BODY_IC_VEC deserialize_body_ic_vec_from_csv(std::istream &);
    BODY_IC_VEC deserialize_body_ic_vec_from_csv(const std::string &);

    /// BINARY
    /// - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
    /// - second 4 bytes: number of bodies
    /// - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_IC
    /// - no separator, no newline, binary

    void serialize_body_ic_vec_to_bin(std::ostream &, const BODY_IC_VEC &);
    void serialize_body_ic_vec_to_bin(const std::string &, const BODY_IC_VEC &);

    BODY_IC_VEC deserialize_body_ic_vec_from_bin(std::istream &);
    BODY_IC_VEC deserialize_body_ic_vec_from_bin(const std::string &);
}