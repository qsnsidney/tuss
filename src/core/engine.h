#pragma once

#include <vector>
#include <optional>
#include "physics.hpp"

namespace CORE
{
    /// Interface
    class ENGINE
    {
    public:
        ENGINE(CORE::SYSTEM_STATE system_state_ic, CORE::DT dt, std::optional<std::string> system_state_log_dir_opt = {});
        virtual ~ENGINE() = 0;

        // Main entrance
        virtual const CORE::SYSTEM_STATE &run(int n_iter) final;

    protected:
        // To be defined
        // Continues execution from previous SYSTEM_STATE
        virtual CORE::SYSTEM_STATE execute(int n_iter) = 0;

    protected:
        const CORE::SYSTEM_STATE &system_state_snapshot() const { return system_state_snapshot_; }
        CORE::DT dt() const { return dt_; }

        bool is_system_state_logging_enabled() const { return system_state_log_dir_opt_.has_value(); }
        // P signature: CORE::SYSTEM_STATE system_state_producer()
        template <typename P>
        void push_system_state_to_log(P system_state_producer)
        {
            if (is_system_state_logging_enabled())
                push_system_state_to_log(system_state_producer());
        }
        void push_system_state_to_log(CORE::SYSTEM_STATE system_state);
        void serialize_system_state_log();

        int num_logged_iterations() const;

    private:
        void set_system_state_snapshot(CORE::SYSTEM_STATE system_state_snapshot) { system_state_snapshot_ = std::move(system_state_snapshot); }

    private:
        CORE::SYSTEM_STATE system_state_snapshot_;
        CORE::DT dt_;

        std::optional<std::string> system_state_log_dir_opt_;
        std::vector<CORE::SYSTEM_STATE> system_state_log_;
        int num_system_state_log_popped_ = 0;
    };
}
