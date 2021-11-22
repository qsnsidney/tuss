#pragma once

#include <sstream>
#include <functional>
#include <vector>
#include <iostream>

/// Helpers

/*
 * Concatenate preprocessor tokens A and B without expanding macro definitions
 * (however, if invoked from a macro, macro arguments are expanded).
 */
#define UTST_PPCAT_NX(A, B) A##B

/*
 * Concatenate preprocessor tokens A and B after macro-expanding them.
 */
#define UTST_PPCAT(A, B) UTST_PPCAT_NX(A, B)

/*
 * Turn A into a string literal without expanding macro definitions
 * (however, if invoked from a macro, macro arguments are expanded).
 */
#define UTST_STRINGIZE_NX(A) #A

/*
 * Turn A into a string literal after macro-expanding it.
 */
#define UTST_STRINGIZE(A) UTST_STRINGIZE_NX(A)

namespace CORE::UTST
{
    class TEST_REGISTRY
    {
    public:
        using function_type = std::function<void()>;

        void register_function(std::string f_name, function_type f)
        {
            registered_functions_.emplace_back(std::move(f_name), std::move(f));
        }

        void execute_functions() const
        {
            for (const auto &[f_name, f] : registered_functions_)
            {
                std::cout << "Running test [" << f_name << "] .." << std::endl;
                f();
            }
        }

    private:
        std::vector<std::pair<std::string, function_type> > registered_functions_;
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

#define UTST_TEST(test_name)                                                                                                    \
    static void UTST_PPCAT(__utst_test_function_, test_name)();                                                                 \
    static struct UTST_PPCAT(__utst_test_register_struct_, test_name)                                                           \
    {                                                                                                                           \
        UTST_PPCAT(__utst_test_register_struct_, test_name)                                                                     \
        ()                                                                                                                      \
        {                                                                                                                       \
            __utst_test_registry.register_function(UTST_STRINGIZE_NX(test_name), UTST_PPCAT(__utst_test_function_, test_name)); \
        }                                                                                                                       \
    } UTST_PPCAT(__utst_test_register_struct_inst_, test_name);                                                                 \
    static void UTST_PPCAT(__utst_test_function_, test_name)()

#define UTST_ASSERT(condition)                                                                                                                              \
    {                                                                                                                                                       \
        if (!(condition))                                                                                                                                   \
        {                                                                                                                                                   \
            std::string msg = std::string("\nASSERT ") + UTST_STRINGIZE(condition) + "\n" +                                                                 \
                              std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") + std::string(__PRETTY_FUNCTION__); \
            throw std::runtime_error(msg);                                                                                                                  \
        }                                                                                                                                                   \
    }

#define UTST_ASSERT_EQUAL(x, y)                                                                                                                             \
    {                                                                                                                                                       \
        if ((x) != (y))                                                                                                                                     \
        {                                                                                                                                                   \
            std::stringstream x_sstream;                                                                                                                    \
            x_sstream << (x);                                                                                                                               \
            std::stringstream y_sstream;                                                                                                                    \
            y_sstream << (y);                                                                                                                               \
            std::string msg = std::string("\nASSERT ") + x_sstream.str() + std::string(" != ") + y_sstream.str() + "\n" +                                   \
                              std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") + std::string(__PRETTY_FUNCTION__); \
            throw std::runtime_error(msg);                                                                                                                  \
        }                                                                                                                                                   \
    }
