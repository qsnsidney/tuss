#include "buffer.h"

namespace CPUSIM
{
    std::ostream &operator<<(std::ostream &os, const BUFFER &buf)
    {
        int counter = 0;
        os << "POS: ";
        for (auto p : buf.pos)
        {
            os << "[" << counter << "] " << p << ", ";
            counter++;
        }
        os << std::endl;

        counter = 0;
        os << "VEL: ";
        for (auto v : buf.vel)
        {
            os << "[" << counter << "] " << v << ", ";
            counter++;
        }
        os << std::endl;

        counter = 0;
        os << "ACC: ";
        for (auto a : buf.acc)
        {
            os << "[" << counter << "] " << a << ", ";
            counter++;
        }
        os << std::endl;

        return os;
    }

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

    void debug_workspace(const BUFFER &buffer, const std::vector<CORE::MASS> &mass)
    {
        int counter = 0;
        std::cout << "MASS: ";
        for (auto m : mass)
        {
            std::cout << "[" << counter << "] " << m << ", ";
            counter++;
        }
        std::cout << std::endl;

        std::cout << buffer;

        std::cout << std::endl;
    }
}