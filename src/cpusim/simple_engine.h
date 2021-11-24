#pragma once

#include "engine.h"

namespace CPUSIM
{
    class SIMPLE_ENGINE final : public ENGINE
    {
    public:
        virtual ~SIMPLE_ENGINE() = default;

        using ENGINE::ENGINE;

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;
    };
}