#include "engine.h"
#include "serde.h"

namespace CORE
{
    ENGINE::ENGINE(
        CORE::BODY_STATE_VEC body_states_ic,
        CORE::DT dt,
        std::optional<std::string> body_states_log_dir_opt) : body_states_ic_(body_states_ic),
                                                              dt_(dt),
                                                              body_states_log_dir_opt_(body_states_log_dir_opt)
    {
    }

    ENGINE::~ENGINE()
    {
        serialize_body_states_log();
    }

    void ENGINE::run(int n_iter)
    {
        set_body_states_ic(execute(n_iter));
    }

    void ENGINE::set_body_states_ic(CORE::BODY_STATE_VEC body_states_ic)
    {
        body_states_ic_ = std::move(body_states_ic);
    }

    void ENGINE::push_body_states_to_log(CORE::BODY_STATE_VEC body_states)
    {
        if (!is_body_states_logging_enabled())
        {
            return;
        }
        body_states_log_.emplace_back(std::move(body_states));
    }

    void ENGINE::serialize_body_states_log()
    {
        if (!is_body_states_logging_enabled())
        {
            return;
        }

        for (unsigned i = 0; i < body_states_log_.size(); i++)
        {
            unsigned body_states_log_id = i + num_body_states_log_popped_;
            std::string filename = *body_states_log_dir_opt_ + "/" + std::to_string(body_states_log_id) + ".bin";
            CORE::serialize_body_state_vec_to_bin(filename, body_states_log_[i]);
        }

        // Clear the log
        num_body_states_log_popped_ += body_states_log_.size();
        body_states_log_.clear();
    }

    int ENGINE::num_logged_iterations() const
    {
        return num_body_states_log_popped_ + body_states_log_.size();
    }
}