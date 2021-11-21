#pragma once

#include "physics.h"

namespace CPUSIM
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
