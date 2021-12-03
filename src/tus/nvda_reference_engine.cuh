#pragma once

#include "core/engine.h"

namespace TUS
{
    class NVDA_REFERENCE_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~NVDA_REFERENCE_ENGINE() = default;

        NVDA_REFERENCE_ENGINE(CORE::SYSTEM_STATE body_states_ic,
                         CORE::DT dt,
                         int block_size,
                         std::optional<std::string> system_state_log_dir_opt = {});

        virtual CORE::SYSTEM_STATE execute(int n_iter) override;

    private:
        int block_size_;
    };
}