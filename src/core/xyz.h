#pragma once

#include <ostream>

namespace CORE
{
    /// XYZ_BASE type
    template <typename T>
    struct XYZ_BASE
    {
        using TYPE = T;

        T x;
        T y;
        T z;
    };

    /// Use this type
    using XYZ = XYZ_BASE<float>;

    /// Operators

    template <typename T>
    bool operator==(const XYZ_BASE<T> &lhs, const XYZ_BASE<T> &rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }

    template <typename T>
    bool operator!=(const XYZ_BASE<T> &lhs, const XYZ_BASE<T> &rhs)
    {
        return !(lhs == rhs);
    }

    template <typename T>
    XYZ_BASE<T> &operator+=(XYZ_BASE<T> &lhs, const XYZ_BASE<T> &rhs)
    {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
        return lhs;
    }

    template <typename T>
    XYZ_BASE<T> operator+(const XYZ_BASE<T> &lhs, const XYZ_BASE<T> &rhs)
    {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }

    template <typename T>
    XYZ_BASE<T> &operator-=(XYZ_BASE<T> &lhs, const XYZ_BASE<T> &rhs)
    {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
        return lhs;
    }

    template <typename T>
    XYZ_BASE<T> operator-(const XYZ_BASE<T> &lhs, const XYZ_BASE<T> &rhs)
    {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }

    template <typename T>
    XYZ_BASE<T> operator-(const XYZ_BASE<T> &data)
    {
        return {-data.x, -data.y, -data.z};
    }

    template <typename T>
    XYZ_BASE<T> &operator*=(XYZ_BASE<T> &lhs, T multiplier)
    {
        lhs.x *= multiplier;
        lhs.y *= multiplier;
        lhs.z *= multiplier;
        return lhs;
    }

    template <typename T>
    XYZ_BASE<T> operator*(const XYZ_BASE<T> &lhs, T multiplier)
    {
        return {lhs.x * multiplier, lhs.y * multiplier, lhs.z * multiplier};
    }

    template <typename T>
    XYZ_BASE<T> operator*(T multiplier, const XYZ_BASE<T> &rhs)
    {
        return {rhs.x * multiplier, rhs.y * multiplier, rhs.z * multiplier};
    }

    template <typename T>
    XYZ_BASE<T> &operator/=(XYZ_BASE<T> &lhs, T divisor)
    {
        lhs.x /= divisor;
        lhs.y /= divisor;
        lhs.z /= divisor;
        return lhs;
    }

    template <typename T>
    XYZ_BASE<T> operator/(const XYZ_BASE<T> &lhs, T divisor)
    {
        return {lhs.x / divisor, lhs.y / divisor, lhs.z / divisor};
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const XYZ_BASE<T> &data)
    {
        os << '(' << data.x << ',' << data.y << ',' << data.z << ')';
        return os;
    }
}
