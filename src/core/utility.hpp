#pragma once

#include <utility>
#include <cmath>

namespace CORE
{
    /// https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
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

    inline int linearize_upper_triangle_matrix_index(int i, int j, int n)
    {
        return (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
    }

    inline std::pair<int, int> delinearize_upper_triangle_matrix_index(int k, int n)
    {
        int i = n - 2 - static_cast<int>(std::floor(std::sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5));
        int j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
        return {i, j};
    }
}