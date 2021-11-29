#pragma once

#include <string>

namespace CORE
{
    /// Time stamp function in milliseconds
    double get_time_stamp();

    class TIMER
    {
    public:
        enum class TRIGGER_LEVEL
        {
            IMP,
            INFO,
            DBG
        };
        static void set_trigger_level(TRIGGER_LEVEL trigger_level) { s_trigger_level = trigger_level; }

    public:
        explicit TIMER(std::string profile_name);
        ~TIMER();

        /// Use this to measure time elapsed from preivous call to "elapsed_previous()"
        /// If the trigger_level does not match, the current timepoint will not be saved
        double elapsed_previous(const std::string &subprofile_name, TRIGGER_LEVEL trigger_level = TRIGGER_LEVEL::IMP);

        /// Use this to measure the time elapsed from the creation of this TIMER instance
        double elapsed_start() const;

    private:
        double start_time_;
        double previous_elapsing_time_;
        std::string profile_name_;

    private:
        static bool match_trigger_level(TRIGGER_LEVEL trigger_level) { return s_trigger_level >= trigger_level; }
        static TRIGGER_LEVEL s_trigger_level;
    };
}
