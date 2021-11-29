#include "timer.h"

#include <iostream>
#include <sys/time.h>

namespace CORE
{
    double get_time_stamp()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (double)tv.tv_usec / 1000000 + tv.tv_sec;
    }

    TIMER::VERBOSITY TIMER::s_verbosity = TIMER::VERBOSITY::IMP;

    TIMER::TIMER(std::string profile_name) : start_time_(get_time_stamp()), previous_elapsing_time_(start_time_), profile_name_(std::move(profile_name))
    {
        std::cout << "TIMER: Starting profile (" << profile_name_ << ")" << std::endl;
    }

    TIMER::~TIMER()
    {
        elapsed_start();
    }

    double TIMER::elapsed_previous(const std::string &subprofile_name, VERBOSITY verbosity)
    {

        double current_time = get_time_stamp();
        double elapsed = current_time - previous_elapsing_time_;

        if (match_verbosity(verbosity))
        {
            std::cout << "TIMER: Subprofile [" << profile_name_ << "/" << subprofile_name
                      << "]: " << std::to_string(elapsed) << " seconds" << std::endl;
        }

        previous_elapsing_time_ = current_time;
        return elapsed;
    }

    double TIMER::elapsed_start() const
    {
        double current_time = get_time_stamp();
        double elapsed = current_time - start_time_;

        std::cout << "TIMER: Profile [" << profile_name_ << "]: " << std::to_string(elapsed) << " seconds" << std::endl;

        return elapsed;
    }
}