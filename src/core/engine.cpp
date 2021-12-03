#include "engine.h"
#include "serde.h"

namespace CORE
{
    ENGINE::ENGINE(
        CORE::SYSTEM_STATE system_state_ic,
        CORE::DT dt,
        std::optional<std::string> system_state_log_dir_opt) : system_state_snapshot_(std::move(system_state_ic)),
                                                               dt_(dt),
                                                               system_state_log_dir_opt_(system_state_log_dir_opt)
    {
    }

    ENGINE::~ENGINE()
    {
        serialize_system_state_log();
    }

    const CORE::SYSTEM_STATE &ENGINE::run(int n_iter)
    {
        set_system_state_snapshot(execute(n_iter));
        return system_state_snapshot();
    }

    void ENGINE::push_system_state_to_log(CORE::SYSTEM_STATE system_state)
    {
        if (!is_system_state_logging_enabled())
        {
            return;
        }
        system_state_log_.emplace_back(std::move(system_state));
    }

    void ENGINE::serialize_system_state_log()
    {
        if (!is_system_state_logging_enabled())
        {
            return;
        }

        for (unsigned i = 0; i < system_state_log_.size(); i++)
        {
            unsigned system_state_log_id = i + num_system_state_log_popped_;
            std::string filename = *system_state_log_dir_opt_ + "/" + std::to_string(system_state_log_id) + ".bin";
            CORE::serialize_system_state_to_bin(filename, system_state_log_[i]);
        }

        // Clear the log
        num_system_state_log_popped_ += system_state_log_.size();
        system_state_log_.clear();
    }

    int ENGINE::num_logged_iterations() const
    {
        return num_system_state_log_popped_ + system_state_log_.size();
    }
}