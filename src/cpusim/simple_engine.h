#pragma once

#include "engine.h"

namespace CPUSIM
{
    class SIMPLE_ENGINE final : public ENGINE
    {
    public:
        virtual ~SIMPLE_ENGINE() = default;

        virtual void execute(int n_iter) override;
    };
}