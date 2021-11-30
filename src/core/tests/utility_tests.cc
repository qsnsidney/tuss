#include "utst.hpp"
#include "utility.hpp"

using namespace CORE;

UTST_MAIN();

UTST_TEST(linearize_upper_triangle_matrix_index)
{
    constexpr int n = 5;
    UTST_ASSERT_EQUAL(0, linearize_upper_triangle_matrix_index(0, 1, n));
    UTST_ASSERT_EQUAL(1, linearize_upper_triangle_matrix_index(0, 2, n));
    UTST_ASSERT_EQUAL(2, linearize_upper_triangle_matrix_index(0, 3, n));
    UTST_ASSERT_EQUAL(3, linearize_upper_triangle_matrix_index(0, 4, n));
    UTST_ASSERT_EQUAL(4, linearize_upper_triangle_matrix_index(1, 2, n));
    UTST_ASSERT_EQUAL(5, linearize_upper_triangle_matrix_index(1, 3, n));
    UTST_ASSERT_EQUAL(6, linearize_upper_triangle_matrix_index(1, 4, n));
    UTST_ASSERT_EQUAL(7, linearize_upper_triangle_matrix_index(2, 3, n));
    UTST_ASSERT_EQUAL(8, linearize_upper_triangle_matrix_index(2, 4, n));
    UTST_ASSERT_EQUAL(9, linearize_upper_triangle_matrix_index(3, 4, n));
}

UTST_TEST(delinearize_upper_triangle_matrix_index)
{
    constexpr int n = 5;
    UTST_ASSERT(std::make_pair(0, 1) == delinearize_upper_triangle_matrix_index(0, n));
    UTST_ASSERT(std::make_pair(0, 2) == delinearize_upper_triangle_matrix_index(1, n));
    UTST_ASSERT(std::make_pair(0, 3) == delinearize_upper_triangle_matrix_index(2, n));
    UTST_ASSERT(std::make_pair(0, 4) == delinearize_upper_triangle_matrix_index(3, n));
    UTST_ASSERT(std::make_pair(1, 2) == delinearize_upper_triangle_matrix_index(4, n));
    UTST_ASSERT(std::make_pair(1, 3) == delinearize_upper_triangle_matrix_index(5, n));
    UTST_ASSERT(std::make_pair(1, 4) == delinearize_upper_triangle_matrix_index(6, n));
    UTST_ASSERT(std::make_pair(2, 3) == delinearize_upper_triangle_matrix_index(7, n));
    UTST_ASSERT(std::make_pair(2, 4) == delinearize_upper_triangle_matrix_index(8, n));
    UTST_ASSERT(std::make_pair(3, 4) == delinearize_upper_triangle_matrix_index(9, n));
}
