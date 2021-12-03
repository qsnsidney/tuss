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

    CORE::SYSTEM_STATE generate_system_state(const BUFFER &buffer, const std::vector<CORE::MASS> &mass)
    {
        CORE::SYSTEM_STATE system_state;
        system_state.reserve(mass.size());
        for (size_t i_body = 0; i_body < mass.size(); i_body++)
        {
            system_state.emplace_back(buffer.pos[i_body], buffer.vel[i_body], mass[i_body]);
        }
        return system_state;
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