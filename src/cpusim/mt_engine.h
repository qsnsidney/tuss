#pragma once

#include "core/engine.h"

namespace CPUSIM
{
    class MT_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~MT_ENGINE() = default;

        using ENGINE::ENGINE;

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;
    };
}