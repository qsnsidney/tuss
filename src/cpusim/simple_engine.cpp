#include "simple_engine.h"

namespace
{
    struct BUFFER
    {
        std::vector<CORE::POS> pos;
        std::vector<CORE::VEL> vel;
        std::vector<CORE::ACC> acc;
        std::vector<CORE::MASS> mass;
    };
}

namespace CPUSIM
{
    void SIMPLE_ENGINE::execute(int n_iter)
    {
        BUFFER buf_in;
        BUFFER buf_out;

        // Step 1: Prepare IC
        for (const auto &[body_pos, body_vel, body_mass] : body_ics_)
        {
            buf_in.pos.push_back(body_pos);
            buf_in.vel.push_back(body_vel);
            buf_in.acc.push_back(CORE::ACC());
            buf_in.mass.push_back(body_mass);
        }

        // Step 2
        static_cast<void>(dt_);
    }
}