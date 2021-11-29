#pragma once

#include "core/engine.h"
#include <optional>
#include "threading.h"

namespace CPUSIM
{
    class MT_ENGINE final : public CORE::ENGINE
    {
    public:
        virtual ~MT_ENGINE() = default;

        MT_ENGINE(CORE::BODY_STATE_VEC body_states_ic,
                  CORE::DT dt,
                  size_t n_thread,
                  bool use_thread_pool,
                  std::optional<std::string> body_states_log_dir_opt = {});

        virtual CORE::BODY_STATE_VEC execute(int n_iter) override;

    private:
        template <typename Function>
        void parallel_for_helper(int begin, int end, Function &&f)
        {
            if (thread_pool_opt_)
            {
                parallel_for(*thread_pool_opt_, begin, end, std::forward<Function>(f));
            }
            else
            {
                parallel_for(n_thread_, begin, end, std::forward<Function>(f));
            }
        }

        size_t n_thread_;
        std::optional<THREAD_POOL> thread_pool_opt_ = std::nullopt;
    };
}