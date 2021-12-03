#pragma once

#include <iostream>

#include "physics.hpp"

namespace CORE
{
    /// CSV
    /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row

    void serialize_system_state_to_csv(std::ostream &, const SYSTEM_STATE &);
    void serialize_system_state_to_csv(const std::string &, const SYSTEM_STATE &);

    SYSTEM_STATE deserialize_system_state_from_csv(std::istream &);
    SYSTEM_STATE deserialize_system_state_from_csv(const std::string &);

    /// BINARY
    /// - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
    /// - second 4 bytes: number of bodies
    /// - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_STATE
    /// Everything in binary

    void serialize_system_state_to_bin(std::ostream &, const SYSTEM_STATE &);
    void serialize_system_state_to_bin(const std::string &, const SYSTEM_STATE &);

    SYSTEM_STATE deserialize_system_state_from_bin(std::istream &);
    SYSTEM_STATE deserialize_system_state_from_bin(const std::string &);

    /// Useful
    SYSTEM_STATE deserialize_system_state_from_file(const std::string &);
}