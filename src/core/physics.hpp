#pragma once

#include <tuple>
#include <vector>
#include <cmath>
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

    /// Implementations

    inline ACC ACC::from_gravity(const POS &p_src, MASS m_src, const POS &p_target)
    {
        /// TODO: utst
        const XYZ displacement = p_src - p_target;
        const UNIVERSE::floating_value_type denom = std::pow(displacement.norm_square() + UNIVERSE::epislon_square, -1.5f);

        return {m_src * displacement * denom};
    }

    inline VEL VEL::updated(const VEL &v, const ACC &a, DT dt)
    {
        return {v + static_cast<UNIVERSE::floating_value_type>(0.5) * a * dt};
    }

    inline POS POS::updated(const POS &p, const VEL &v, const ACC &a, DT dt)
    {
        return {p + v * dt + static_cast<UNIVERSE::floating_value_type>(0.5) * a * dt * dt};
    }
}
