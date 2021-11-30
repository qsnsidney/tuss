#pragma once

#include "core/engine.h"

namespace CPUSIM
{
    class SHARED_ACC_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~SHARED_ACC_ENGINE() = default;

        using ENGINE::ENGINE;

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;
    };
}