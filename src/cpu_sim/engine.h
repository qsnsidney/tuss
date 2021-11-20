#pragma once

#include "physics.h"

namespace CPU_SIM
{
    /// Interface
    class ENGINE
    {
    public:
        virtual ~ENGINE() = 0;

    public:
        virtual void execute() = 0;
    };
}
