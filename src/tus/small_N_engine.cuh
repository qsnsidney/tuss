#pragma once

#include "core/engine.h"

namespace TUS
{
    class SMALL_N_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~SMALL_N_ENGINE() = default;

        SMALL_N_ENGINE(CORE::SYSTEM_STATE body_states_ic,
                              CORE::DT dt,
                              int block_size,
                              int tb_len,
                              int tb_wid,
                              int unroll_factor,
                              int tpb,
                              std::optional<std::string> system_state_log_dir_opt = {});

        virtual std::string name() override { return "SMALL_N_ENGINE"; }
        virtual CORE::SYSTEM_STATE execute(int n_iter, CORE::TIMER &timer) override;

    private:
        int block_size_;
        int tb_len_;
        int tb_wid_;
        int unroll_factor_;
        int tpb_;
    };
}