#pragma once

#include "physics.h"
#include <ostream>

namespace CORE
{
    /// CSV
    /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row

    std::vector<BODY_IC> parse_body_ic_from_csv(std::istream &);
    std::vector<BODY_IC> parse_body_ic_from_csv(const std::string &);

    /// BINARY
    /// - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
    /// - second 4 bytes: number of bodies
    /// - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_IC
    /// - no separator, no newline, binary

    void serialize_body_ic_to_bin(std::ostream &, const std::vector<BODY_IC> &);
    void serialize_body_ic_to_bin(const std::string &, const std::vector<BODY_IC> &);

    std::vector<BODY_IC> deserialize_body_ic_from_bin(std::istream &);
    std::vector<BODY_IC> deserialize_body_ic_from_bin(const std::string &);
}