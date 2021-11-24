#pragma once

#include <tuple>
#include <vector>
#include "xyz.hpp"
#include "universe.hpp"

namespace CORE
{
    /// Basic types

    using DT = UNIVERSE::floating_value_type;

    using MASS = UNIVERSE::floating_value_type;

    struct ACC;
    struct VEL;
    struct POS;

    struct ACC : public XYZ
    {
        static ACC from_gravity(const POS &p_src, MASS m_src, const POS &p_target);
    };

    struct VEL : public XYZ
    {
        static VEL updated(const VEL &, const ACC &, DT);
    };
    struct POS : public XYZ
    {
        static POS updated(const POS &, const VEL &, const ACC &, DT);
    };

    /// Input/output types

    using BODY_STATE = std::tuple<POS, VEL, MASS>;
    using BODY_STATE_VEC = std::vector<BODY_STATE>;
}
