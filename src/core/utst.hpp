#pragma once

#include "macros.hpp"
#include <sstream>
#include <functional>
#include <vector>
#include <iostream>
#include <tuple>

/// Helpers

namespace CORE::UTST
{
    class TEST_REGISTRY
    {
    public:
        using function_type = std::function<void()>;

        void register_active_function(std::string f_name, function_type f)
        {
            registered_functions_.emplace_back(std::move(f_name), std::move(f), true);
        }

        void register_inactive_function(std::string f_name, function_type f)
        {
            registered_functions_.emplace_back(std::move(f_name), std::move(f), false);
        }

        void execute_functions() const
        {
            for (const auto &[f_name, f, is_active] : registered_functions_)
            {
                if (is_active)
                {
                    std::cout << "Running test [" << f_name << "] .." << std::endl;
                    f();
                }
                else
                {
                    std::cout << "Not running test [" << f_name << "] .." << std::endl;
                }
            }
        }

    private:
        std::vector<std::tuple<std::string, function_type, bool> > registered_functions_;
    };
}

/// UTST Utilities

#define UTST_MAIN()                                        \
    static CORE::UTST::TEST_REGISTRY __utst_test_registry; \
                                                           \
    int main()                                             \
    {                                                      \
        __utst_test_registry.execute_functions();          \
    }

#define UTST_TEST(test_name)                                                                                                 \
    static void PPCAT(__utst_test_function_, test_name)();                                                                   \
    static struct PPCAT(__utst_test_register_struct_, test_name)                                                             \
    {                                                                                                                        \
        PPCAT(__utst_test_register_struct_, test_name)                                                                       \
        ()                                                                                                                   \
        {                                                                                                                    \
            __utst_test_registry.register_active_function(STRINGIZE_NX(test_name), PPCAT(__utst_test_function_, test_name)); \
        }                                                                                                                    \
    } PPCAT(__utst_test_register_struct_inst_, test_name);                                                                   \
    static void PPCAT(__utst_test_function_, test_name)()

#define UTST_IGNORED_TEST(test_name)                                                                                           \
    static void PPCAT(__utst_test_function_, test_name)();                                                                     \
    static struct PPCAT(__utst_test_register_struct_, test_name)                                                               \
    {                                                                                                                          \
        PPCAT(__utst_test_register_struct_, test_name)                                                                         \
        ()                                                                                                                     \
        {                                                                                                                      \
            __utst_test_registry.register_inactive_function(STRINGIZE_NX(test_name), PPCAT(__utst_test_function_, test_name)); \
        }                                                                                                                      \
    } PPCAT(__utst_test_register_struct_inst_, test_name);                                                                     \
    static void PPCAT(__utst_test_function_, test_name)()

#define UTST_ASSERT(condition)                                                                                                                                                                        \
    {                                                                                                                                                                                                 \
        if (!(condition))                                                                                                                                                                             \
        {                                                                                                                                                                                             \
            std::string msg = std::string("\nUTST_ASSERT ") + STRINGIZE(condition) + "\n" +                                                                                                           \
                                                                        std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") + std::string(__PRETTY_FUNCTION__); \
            throw std::runtime_error(msg);                                                                                                                                                            \
        }                                                                                                                                                                                             \
    }

#define UTST_ASSERT_EQUAL(x, y)                                                                                                                             \
    {                                                                                                                                                       \
        if ((x) != (y))                                                                                                                                     \
        {                                                                                                                                                   \
            std::stringstream x_sstream;                                                                                                                    \
            x_sstream << (x);                                                                                                                               \
            std::stringstream y_sstream;                                                                                                                    \
            y_sstream << (y);                                                                                                                               \
            std::string msg = std::string("\nUTST_ASSERT ") + x_sstream.str() + std::string(" != ") + y_sstream.str() + "\n" +                              \
                              std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") + std::string(__PRETTY_FUNCTION__); \
            throw std::runtime_error(msg);                                                                                                                  \
        }                                                                                                                                                   \
    }
