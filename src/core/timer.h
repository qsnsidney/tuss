#pragma once

#include <sys/time.h>

namespace CORE
{
    /// time stamp function in milliseconds
    double get_time_stamp()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (double)tv.tv_usec / 1000000 + tv.tv_sec;
    }
}
