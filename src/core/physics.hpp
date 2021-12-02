#pragma once

#include <tuple>
#include <vector>
#include <cmath>
#include <iostream>
#include "xyz.hpp"
#include "universe.hpp"
#include "macros.hpp"

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

    /// A field caused by p_src to p_target, a vector pointing from p_target to p_src
    XYZ universal_field(const POS &p_src, const POS &p_target);

    /// Input/output types

    using BODY_STATE = std::tuple<POS, VEL, MASS>;
    using BODY_STATE_VEC = std::vector<BODY_STATE>;

    /// Comparison
    bool verify(const BODY_STATE_VEC &expected_state_vec, const BODY_STATE_VEC &actual_state_vec);

    /// Implementations

    inline ACC ACC::from_gravity(const POS &p_src, MASS m_src, const POS &p_target)
    {
        return {m_src * universal_field(p_src, p_target)};
    }

    inline VEL VEL::updated(const VEL &v, const ACC &a, DT dt)
    {
        return {v + static_cast<UNIVERSE::floating_value_type>(0.5) * a * dt};
    }

    inline POS POS::updated(const POS &p, const VEL &v, const ACC &a, DT dt)
    {
        return {p + v * dt + static_cast<UNIVERSE::floating_value_type>(0.5) * a * dt * dt};
    }

    inline XYZ universal_field(const POS &p_src, const POS &p_target)
    {
        const XYZ displacement = p_src - p_target;
        const UNIVERSE::floating_value_type denom_base = displacement.norm_square() + UNIVERSE::epislon_square;

        return displacement / (denom_base * std::sqrt(denom_base));
    }

    inline bool verify(const BODY_STATE_VEC &expected_state_vec, const BODY_STATE_VEC &actual_state_vec)
    {
        ASSERT(expected_state_vec.size() == actual_state_vec.size());
        const size_t n_body = expected_state_vec.size();

        for (size_t i_body = 0; i_body < n_body; i_body++)
        {
            // Mass must match exactly
            if (std::get<MASS>(expected_state_vec[i_body]) != std::get<MASS>(actual_state_vec[i_body]))
            {
                std::cout << "body " << i_body << ": "
                          << " expected_mass " << std::get<MASS>(expected_state_vec[i_body])
                          << " does not match with actual_mass " << std::get<MASS>(actual_state_vec[i_body]) << std::endl;
                ASSERT(false);
            }

            const auto pos_err_square = (std::get<POS>(expected_state_vec[i_body]) - std::get<POS>(actual_state_vec[i_body])).norm_square();
            if (pos_err_square > UNIVERSE::epislon_square)
            {
                std::cout << "body " << i_body << ": "
                          << "error_square of POS  " << pos_err_square
                          << " is larger than acceptance " << UNIVERSE::epislon_square << std::endl;
                return false;
            }

            const auto vel_err_square = (std::get<VEL>(expected_state_vec[i_body]) - std::get<VEL>(actual_state_vec[i_body])).norm_square();
            if (vel_err_square > UNIVERSE::epislon_square)
            {
                std::cout << "body " << i_body << ": "
                          << "error_square of VEL  " << vel_err_square
                          << " is larger than acceptance " << UNIVERSE::epislon_square << std::endl;
                return false;
            }
        }

        return true;
    }
}
