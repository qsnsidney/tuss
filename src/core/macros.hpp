#pragma once

/*
 * Concatenate preprocessor tokens A and B without expanding macro definitions
 * (however, if invoked from a macro, macro arguments are expanded).
 */
#define PPCAT_NX(A, B) A##B

/*
 * Concatenate preprocessor tokens A and B after macro-expanding them.
 */
#define PPCAT(A, B) PPCAT_NX(A, B)

/*
 * Turn A into a string literal without expanding macro definitions
 * (however, if invoked from a macro, macro arguments are expanded).
 */
#define STRINGIZE_NX(A) #A

/*
 * Turn A into a string literal after macro-expanding it.
 */
#define STRINGIZE(A) STRINGIZE_NX(A)

#define ASSERT(condition)                                                                                                                                                          \
    {                                                                                                                                                                              \
        if (!(condition))                                                                                                                                                          \
        {                                                                                                                                                                          \
            std::string msg =                                                                                                                                                      \
                std::string("\nASSERT ") + STRINGIZE(condition) + "\n" +                                                                                                           \
                                                     std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" in ") + std::string(__PRETTY_FUNCTION__); \
            throw std::runtime_error(msg);                                                                                                                                         \
        }                                                                                                                                                                          \
    }
