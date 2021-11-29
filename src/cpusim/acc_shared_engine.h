#pragma once

#include "core/engine.h"

namespace CPUSIM
{
    class ACC_SHARED_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~ACC_SHARED_ENGINE() = default;

        using ENGINE::ENGINE;

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;
    };
}