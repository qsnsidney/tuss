#pragma once

#include <utility>
#include <cmath>

namespace CORE
{
    /// https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    /// WARNING: Watch for potential overflowing
    /// n: width and length
    /// # non-zero elem: n * (n - 1) / 2
    /// k:
    ///   j  0   1   2   3   4
    /// i   --  --  --  --  --
    /// 0 |  0  a0  a1  a2  a3
    /// 1 |  0  00  a4  a5  a6
    /// 2 |  0   0   0  a7  a8
    /// 3 |  0   0   0   0  a9
    /// 4 |  0   0   0   0   0

    inline size_t linearize_upper_triangle_matrix_index(size_t i, size_t j, size_t n)
    {
        return (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
    }

    inline std::pair<size_t, size_t> delinearize_upper_triangle_matrix_index(size_t k, size_t n)
    {
        size_t i = n - 2 - static_cast<size_t>(std::floor(std::sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5));
        size_t j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
        return {i, j};
    }
}