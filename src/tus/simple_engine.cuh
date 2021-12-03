#pragma once

#include "core/engine.h"

namespace TUS
{
    class SIMPLE_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~SIMPLE_ENGINE() = default;

        SIMPLE_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                      CORE::DT dt,
                      int block_size,
                      std::optional<std::string> system_state_log_dir_opt = {});

        virtual std::string name() override { return "SIMPLE_ENGINE"; }
        virtual CORE::SYSTEM_STATE execute(int n_iter, CORE::TIMER &timer) override;

    private:
        int block_size_;
    };
}