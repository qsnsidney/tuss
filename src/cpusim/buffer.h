#pragma once

#include <vector>
#include <iostream>
#include "core/physics.hpp"

namespace CPUSIM
{
    struct BUFFER
    {
        std::vector<CORE::POS> pos;
        std::vector<CORE::VEL> vel;
        std::vector<CORE::ACC> acc;

        BUFFER(int n_body) : pos(n_body, {0, 0, 0}), vel(n_body, {0, 0, 0}), acc(n_body, {0, 0, 0}) {}
    };

    std::ostream &operator<<(std::ostream &os, const BUFFER &buf);

    CORE::SYSTEM_STATE generate_system_state(const BUFFER &buffer, const std::vector<CORE::MASS> &mass);

    void debug_workspace(const BUFFER &buffer, const std::vector<CORE::MASS> &mass);
}