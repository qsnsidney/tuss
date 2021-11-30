#pragma once

#include "basic_engine.h"

namespace CPUSIM
{
    class SHARED_ACC_ENGINE final : public BASIC_ENGINE
    {
    public:
        virtual ~SHARED_ACC_ENGINE() = default;

        using BASIC_ENGINE::BASIC_ENGINE;

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;
    };
}