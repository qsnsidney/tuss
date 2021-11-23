#pragma once

#include <tuple>
#include "xyz.h"
#include "universe.h"

namespace CORE
{
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

    using BODY_IC = std::tuple<POS, VEL, MASS>;
}
