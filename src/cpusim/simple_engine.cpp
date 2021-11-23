#include "simple_engine.h"

namespace
{
    struct BUFFER
    {
        std::vector<CORE::POS> pos;
        std::vector<CORE::VEL> vel;
        std::vector<CORE::ACC> acc;
        std::vector<CORE::MASS> mass;

        BUFFER(int n_body) : pos(n_body, {0, 0, 0}), vel(n_body, {0, 0, 0}), acc(n_body, {0, 0, 0}), mass(n_body, 0) {}
    };
}

namespace CPUSIM
{
    void SIMPLE_ENGINE::execute(CORE::DT dt, int n_iter)
    {
        const int n_body = body_ics().size();

        BUFFER buf_in(n_body);
        BUFFER buf_out(n_body);

        // Step 1: Prepare ic
        for (int i_body = 0; i_body < n_body; i_body++)
        {
            const auto &[body_pos, body_vel, body_mass] = body_ics()[i_body];
            buf_in.pos[i_body] = body_pos;
            buf_in.vel[i_body] = body_vel;
            buf_in.mass[i_body] = body_mass;
        }

        // Step 2: Prepare acceleration for ic
        for (int i_target_body = 0; i_target_body < n_body; i_target_body++)
        {
            for (int j_source_body = 0; j_source_body < n_body; j_source_body++)
            {
                if (i_target_body != j_source_body)
                {
                    buf_in.acc[i_target_body] += CORE::ACC::from_gravity(buf_in.pos[j_source_body], buf_in.mass[j_source_body], buf_in.pos[i_target_body]);
                }
            }
        }
    }
}