#pragma once
#include "engine.h"

namespace CPUSIM
{
    class SIMPLE_ENGINE final : ENGINE
    {
    public:
        virtual ~SIMPLE_ENGINE() = default;

        virtual void execute() override;
    };

}