#pragma once

#include "core/engine.h"

namespace TUS
{
    class SIMPLE_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~SIMPLE_ENGINE() = default;

        SIMPLE_ENGINE(CORE::BODY_STATE_VEC body_states_ic,
                      CORE::DT dt,
                      int block_size,
                      std::optional<std::string> body_states_log_dir_opt = {});

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;

    private:
        int block_size_;
    };
}