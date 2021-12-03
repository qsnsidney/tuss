#pragma once

#include "basic_engine.h"
namespace CPUSIM
{
    class SHARED_ACC_ENGINE final : public BASIC_ENGINE
    {
    public:
        virtual ~SHARED_ACC_ENGINE() = default;

        using BASIC_ENGINE::BASIC_ENGINE;

        virtual CORE::SYSTEM_STATE execute(int n_iter) override;

    private:
        void compute_acceleration(std::vector<CORE::ACC> &acc,
                                    const std::vector<CORE::POS> &pos,
                                    const std::vector<CORE::MASS> &mass);
    };
}