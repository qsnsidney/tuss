#pragma once

#include <vector>
#include "physics.h"

namespace CPUSIM
{
    /// Interface
    class ENGINE
    {
    public:
        virtual ~ENGINE() = 0;

    public:
        virtual void init(CORE::BODY_IC_VEC body_ics) final { body_ics_ = std::move(body_ics); }

        virtual void execute(CORE::DT dt, int n_iter) = 0;

    protected:
        const CORE::BODY_IC_VEC &body_ics() const { return body_ics_; }

    private:
        CORE::BODY_IC_VEC body_ics_;
    };
}
