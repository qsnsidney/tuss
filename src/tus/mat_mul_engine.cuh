#pragma once

#include "core/engine.h"

namespace TUS
{
    class MAT_MUL_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~MAT_MUL_ENGINE() = default;

        MAT_MUL_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                      CORE::DT dt,
                      int block_size,
                      std::optional<std::string> system_state_log_dir_opt = {});

        virtual std::string name() override { return "MAT_MUL_ENGINE"; }
        virtual CORE::SYSTEM_STATE execute(int n_iter, CORE::TIMER &timer) override;

    private:
        int block_size_;
    };
}