#include "physics.h"
#include <cmath>

namespace CORE
{
    ACC ACC::from_gravity(const POS &p_src, MASS m_src, const POS &p_target)
    {
        /// TODO: utst
        const XYZ displacement = p_target - p_src;
        const UNIVERSE::floating_value_type denom = std::pow(displacement.norm_square() + UNIVERSE::epislon_square, -1.5f);

        return {m_src * displacement * denom};
    }

    VEL VEL::updated(const VEL &v, const ACC &a, DT dt)
    {
        return {v + static_cast<UNIVERSE::floating_value_type>(0.5) * a * dt};
    }

    POS POS::updated(const POS &p, const VEL &v, const ACC &a, DT dt)
    {
        return {p + v * dt + static_cast<UNIVERSE::floating_value_type>(0.5) * a * dt * dt};
    }
}