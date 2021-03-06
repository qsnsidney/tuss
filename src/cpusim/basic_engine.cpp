#include "basic_engine.h"
#include "core/timer.h"
#include "buffer.h"
#include "threading.h"

#include <iostream>

namespace CPUSIM
{
    BASIC_ENGINE::BASIC_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                               CORE::DT dt,
                               size_t n_thread,
                               bool use_thread_pool,
                               std::optional<std::string> system_state_log_dir_opt)
        : ENGINE(std::move(system_state_ic), dt, std::move(system_state_log_dir_opt)),
          n_thread_(n_thread),
          thread_pool_opt_(use_thread_pool ? std::make_optional<THREAD_POOL>(n_thread_) : std::nullopt)
    {
        std::cout << "Using " << n_thread << " threads " << (use_thread_pool ? "WITH" : "without") << " threadpool" << std::endl;
    }

    CORE::SYSTEM_STATE BASIC_ENGINE::execute(int n_iter, CORE::TIMER &timer)
    {
        const size_t n_body = system_state_snapshot().size();

        std::vector<CORE::MASS> mass(n_body, 0);
        BUFFER buf_in(n_body);
        // Step 1: Prepare ic
        for (size_t i_body = 0; i_body < n_body; i_body++)
        {
            const auto &[body_pos, body_vel, body_mass] = system_state_snapshot()[i_body];
            buf_in.pos[i_body] = body_pos;
            buf_in.vel[i_body] = body_vel;
            mass[i_body] = body_mass;
        }
        timer.elapsed_previous("step1");

        // Step 2: Prepare acceleration for ic
        parallel_for_helper(0, n_body,
                            [n_body, &buf_in, &mass](size_t i_target_body)
                            {
                                buf_in.acc[i_target_body].reset();
                                for (size_t j_source_body = 0; j_source_body < n_body; j_source_body++)
                                {
                                    if (i_target_body != j_source_body)
                                    {
                                        buf_in.acc[i_target_body] += CORE::ACC::from_gravity(buf_in.pos[j_source_body], mass[j_source_body], buf_in.pos[i_target_body]);
                                    }
                                }
                            });
        timer.elapsed_previous("step2");

        BUFFER buf_out(n_body);
        std::vector<CORE::VEL> vel_tmp(n_body);
        // Core iteration loop
        for (int i_iter = 0; i_iter < n_iter; i_iter++)
        {
            if (false)
            {
                debug_workspace(buf_in, mass);
            }

            parallel_for_helper(0, n_body,
                                [&buf_out, &buf_in, &vel_tmp, this](size_t i_target_body)
                                {
                                    // Step 3: Compute temp velocity
                                    vel_tmp[i_target_body] =
                                        CORE::VEL::updated(buf_in.vel[i_target_body], buf_in.acc[i_target_body], dt());

                                    // Step 4: Update position
                                    buf_out.pos[i_target_body] =
                                        CORE::POS::updated(buf_in.pos[i_target_body], buf_in.vel[i_target_body], buf_in.acc[i_target_body], dt());
                                });

            parallel_for_helper(0, n_body,
                                [n_body, &buf_out, &vel_tmp, &mass, this](size_t i_target_body)
                                {
                                    buf_out.acc[i_target_body].reset();
                                    // Step 5: Compute acceleration
                                    for (size_t j_source_body = 0; j_source_body < n_body; j_source_body++)
                                    {
                                        if (i_target_body != j_source_body)
                                        {
                                            buf_out.acc[i_target_body] += CORE::ACC::from_gravity(buf_out.pos[j_source_body], mass[j_source_body], buf_out.pos[i_target_body]);
                                        }
                                    }

                                    // Step 6: Update velocity
                                    buf_out.vel[i_target_body] = CORE::VEL::updated(vel_tmp[i_target_body], buf_out.acc[i_target_body], dt());
                                });

            // Write SYSTEM_STATE to log
            if (i_iter == 0)
            {
                push_system_state_to_log([&]()
                                         { return generate_system_state(buf_in, mass); });
            }
            push_system_state_to_log([&]()
                                     { return generate_system_state(buf_out, mass); });
            if (i_iter % 10 == 0)
            {
                serialize_system_state_log();
            }

            // Prepare for next iteration
            std::swap(buf_in, buf_out);

            timer.elapsed_previous(std::string("iter") + std::to_string(i_iter), CORE::TIMER::TRIGGER_LEVEL::INFO);
        }

        timer.elapsed_previous("all_iters");

        return generate_system_state(buf_in, mass);
    }
}