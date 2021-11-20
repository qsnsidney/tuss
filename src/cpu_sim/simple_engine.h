#pragma once
#include "engine.h"

namespace CPU_SIM
{
    class SIMPLE_ENGINE final : ENGINE
    {
    public:
        virtual ~SIMPLE_ENGINE() = default;

        virtual void execute() override;
    };

}