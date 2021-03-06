#pragma once

#include "core/engine.h"
#include <optional>
#include "threading.h"

namespace CPUSIM
{
    class BASIC_ENGINE : public CORE::ENGINE
    {
    public:
        virtual ~BASIC_ENGINE() = default;

        BASIC_ENGINE(CORE::SYSTEM_STATE system_state_ic,
                     CORE::DT dt,
                     size_t n_thread,
                     bool use_thread_pool,
                     std::optional<std::string> system_state_log_dir_opt = {});

        virtual std::string name() override { return "BASIC_ENGINE"; }
        virtual CORE::SYSTEM_STATE execute(int n_iter, CORE::TIMER &timer) override;

    protected:
        /// Function signature: void(size_t i)
        ///                     void(size_t i, size_t thread_id)
        template <typename Function>
        void parallel_for_helper(size_t begin, size_t end, Function &&f);

        size_t n_thread() const { return n_thread_; }
        std::optional<THREAD_POOL> &thread_pool_opt() { return thread_pool_opt_; }

    private:
        size_t n_thread_;
        std::optional<THREAD_POOL> thread_pool_opt_ = std::nullopt;
    };

    /// Implementation

    template <typename Function>
    void BASIC_ENGINE::parallel_for_helper(size_t begin, size_t end, Function &&f)
    {
        if (n_thread_ == 1)
        {
            for (size_t i = begin; i < end; i++)
            {
                if constexpr (std::is_invocable_v<Function, size_t, size_t>)
                {
                    f(i, 0);
                }
                else
                {
                    f(i);
                }
            }
        }
        else
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
    }
}