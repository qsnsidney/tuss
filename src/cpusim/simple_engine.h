#pragma once

#include "engine.h"
#include <vector>
#include "physics.h"

namespace CPUSIM
{
    class SIMPLE_ENGINE final : ENGINE
    {
    public:
        SIMPLE_ENGINE(std::vector<CORE::BODY_IC> body_ics, CORE::DT dt) : body_ics_(std::move(body_ics)), dt_(dt) {}

        virtual ~SIMPLE_ENGINE() = default;

        virtual void execute(int n_iter) override;

    private:
        const std::vector<CORE::BODY_IC> body_ics_;
        const CORE::DT dt_;
    };
}