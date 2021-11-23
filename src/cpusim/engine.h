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
        virtual void init(CORE::BODY_STATE_VEC body_states) final { body_states_ = std::move(body_states); }

        virtual void execute(CORE::DT dt, int n_iter) = 0;

    protected:
        const CORE::BODY_STATE_VEC &body_states() const { return body_states_; }

    private:
        CORE::BODY_STATE_VEC body_states_;
    };
}
