#include "simple_engine.h"
#include "timer.h"

#include <iostream>

namespace
{
    struct BUFFER
    {
        std::vector<CORE::POS> pos;
        std::vector<CORE::VEL> vel;
        std::vector<CORE::ACC> acc;

        BUFFER(int n_body) : pos(n_body, {0, 0, 0}), vel(n_body, {0, 0, 0}), acc(n_body, {0, 0, 0}) {}
    };

    CORE::BODY_STATE_VEC generate_body_state_vec(const BUFFER &buffer, const std::vector<CORE::MASS> &mass)
    {
        CORE::BODY_STATE_VEC body_states;
        body_states.reserve(mass.size());
        for (size_t i_body = 0; i_body < mass.size(); i_body++)
        {
            body_states.emplace_back(buffer.pos[i_body], buffer.vel[i_body], mass[i_body]);
        }
        return body_states;
    }
}

namespace CPUSIM
{
    CORE::BODY_STATE_VEC SIMPLE_ENGINE::execute(int n_iter)
    {
        CORE::TIMER timer(std::string("execute(") + std::to_string(dt()) + "*" + std::to_string(n_iter) + ")");
        const int n_body = body_states_ic().size();

        std::vector<CORE::MASS> mass(n_body, 0);
        BUFFER buf_in(n_body);
        // Step 1: Prepare ic
        for (int i_body = 0; i_body < n_body; i_body++)
        {
            const auto &[body_pos, body_vel, body_mass] = body_states_ic()[i_body];
            buf_in.pos[i_body] = body_pos;
            buf_in.vel[i_body] = body_vel;
            mass[i_body] = body_mass;
        }
        timer.elapsed_previous("step1");

        // Step 2: Prepare acceleration for ic
        for (int i_target_body = 0; i_target_body < n_body; i_target_body++)
        {
            for (int j_source_body = 0; j_source_body < n_body; j_source_body++)
            {
                if (i_target_body != j_source_body)
                {
                    buf_in.acc[i_target_body] += CORE::ACC::from_gravity(buf_in.pos[j_source_body], mass[j_source_body], buf_in.pos[i_target_body]);
                }
            }
        }
        timer.elapsed_previous("step2");

        BUFFER buf_out(n_body);
        std::vector<CORE::VEL> vel_tmp(n_body);
        // Core iteration loop
        for (int i_iter = 0; i_iter < n_iter; i_iter++)
        {
            for (int i_target_body = 0; i_target_body < n_body; i_target_body++)
            {
                // Step 3: Compute temp velocity
                vel_tmp[i_target_body] = CORE::VEL::updated(buf_in.vel[i_target_body], buf_in.acc[i_target_body], dt());

                // Step 4: Update position
                buf_out.pos[i_target_body] = CORE::POS::updated(buf_in.pos[i_target_body], buf_in.vel[i_target_body], buf_in.acc[i_target_body], dt());
            }

            for (int i_target_body = 0; i_target_body < n_body; i_target_body++)
            {
                // Step 5: Compute acceleration
                for (int j_source_body = 0; j_source_body < n_body; j_source_body++)
                {
                    if (i_target_body != j_source_body)
                    {
                        buf_out.acc[i_target_body] += CORE::ACC::from_gravity(buf_out.pos[j_source_body], mass[j_source_body], buf_out.pos[i_target_body]);
                    }
                }

                // Step 6: Update velocity
                buf_out.vel[i_target_body] = CORE::VEL::updated(vel_tmp[i_target_body], buf_out.acc[i_target_body], dt());
            }

            // Write BODY_STATE_VEC to log
            if (i_iter == 0)
            {
                push_body_states_to_log([&]()
                                        { return generate_body_state_vec(buf_in, mass); });
            }
            push_body_states_to_log([&]()
                                    { return generate_body_state_vec(buf_out, mass); });

            // Prepare for next iteration
            std::swap(buf_in, buf_out);

            timer.elapsed_previous(std::string("iter") + std::to_string(i_iter));
        }

        return generate_body_state_vec(buf_in, mass);
    }
}