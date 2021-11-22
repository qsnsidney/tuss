#pragma once

#include <string>

namespace CORE
{
    /// Time stamp function in milliseconds
    double get_time_stamp();

    class TIMER
    {
    public:
        explicit TIMER(std::string profile_name);
        ~TIMER();

        /// Use this to measure time elapsed from preivous call to "elapsed_previous()"
        double elapsed_previous(const std::string &subprofile_name);

        /// Use this to measure the time elapsed from the creation of this TIMER instance
        double elapsed_start() const;

    private:
        double start_time_;
        double previous_elapsing_time_;
        std::string profile_name_;
    };
}
