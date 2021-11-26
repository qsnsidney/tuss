#pragma once

#include "core/engine.h"

namespace CPUSIM
{
    class MT_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~MT_ENGINE() = default;

        MT_ENGINE(CORE::BODY_STATE_VEC body_states_ic,
                  CORE::DT dt,
                  int n_thread,
                  std::optional<std::string> body_states_log_dir_opt = {});

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;

    private:
        int n_thread_;
    };
}