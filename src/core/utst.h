#pragma once

#include <sstream>

#define UTST_ASSERT(condition)                                                                                                                                      \
    {                                                                                                                                                               \
        if (!(condition))                                                                                                                                           \
        {                                                                                                                                                           \
            throw std::runtime_error(std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") + std::string(__PRETTY_FUNCTION__)); \
        }                                                                                                                                                           \
    }

#define UTST_ASSERT_EQUAL(x, y)                                                                                                                                                                                                                   \
    {                                                                                                                                                                                                                                             \
        if ((x) != (y))                                                                                                                                                                                                                           \
        {                                                                                                                                                                                                                                         \
            std::stringstream x_sstream;                                                                                                                                                                                                          \
            x_sstream << (x);                                                                                                                                                                                                                     \
            std::stringstream y_sstream;                                                                                                                                                                                                          \
            y_sstream << (y);                                                                                                                                                                                                                     \
            throw std::runtime_error(std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") + std::string(__PRETTY_FUNCTION__) + std::string(": ") + x_sstream.str() + std::string(" != ") + y_sstream.str()); \
        }                                                                                                                                                                                                                                         \
    }
