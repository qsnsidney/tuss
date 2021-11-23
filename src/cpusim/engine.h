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
        virtual void init(std::vector<CORE::BODY_IC> body_ics) final { body_ics_ = std::move(body_ics); }

        virtual void execute(int n_iter) = 0;

    protected:
        const std::vector<CORE::BODY_IC> &body_ics() const { return body_ics_; }

    private:
        std::vector<CORE::BODY_IC> body_ics_;
    };
}
