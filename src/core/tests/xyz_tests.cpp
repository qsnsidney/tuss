#include "utst.h"
#include "xyz.h"

using namespace CORE;

UTST_MAIN();

UTST_TEST(xyz_equal)
{
    XYZ a{1.0, 2.0, 3.0};
    XYZ b{1.0, 2.0, 3.0};
    UTST_ASSERT_EQUAL(a, a);
    UTST_ASSERT_EQUAL(a, b);
}

UTST_TEST(xyz_not_equal)
{
    XYZ a{3.0, 2.0, 1.0};
    XYZ b{1.0, 2.0, 3.0};
    UTST_ASSERT(a != b);
}

UTST_TEST(xyz_add)
{
    XYZ a{1.0, 2.0, 3.0};
    XYZ b{11.0, 12.0, 13.0};
    XYZ res{12.0, 14.0, 16.0};

    XYZ c = a;
    c += b;
    UTST_ASSERT_EQUAL(res, c);

    XYZ d = a + b;
    UTST_ASSERT_EQUAL(res, d);
}

UTST_TEST(xyz_subtract)
{
    XYZ a{7.0, 6.0, 5.0};
    XYZ b{1.0, 6.0, 11.0};
    XYZ res{6.0, 0.0, -6.0};

    XYZ c = a;
    c -= b;
    UTST_ASSERT_EQUAL(res, c);

    XYZ d = a - b;
    UTST_ASSERT_EQUAL(res, d);
}

UTST_TEST(xyz_negative)
{
    XYZ a{0.0, 5.5, -8.9};
    XYZ res{0.0, -5.5, 8.9};

    XYZ c = -a;
    UTST_ASSERT_EQUAL(res, c);
}

UTST_TEST(xyz_multiply)
{
    XYZ a{1.5, 0.0, -3.6};
    float m = -2.7;
    XYZ res{1.5f * m, 0.0, -3.6f * m};

    XYZ b = a;
    b *= m;
    UTST_ASSERT_EQUAL(res, b);

    XYZ c = a * m;
    UTST_ASSERT_EQUAL(res, c);

    XYZ d = m * a;
    UTST_ASSERT_EQUAL(res, d);
}

UTST_TEST(xyz_divide)
{
    XYZ a{2.0, 0.0, -7.9};
    float m = -7.0;
    XYZ res{2.0f / m, 0.0, -7.9f / m};

    XYZ b = a;
    b /= m;
    UTST_ASSERT_EQUAL(res, b);

    XYZ c = a / m;
    UTST_ASSERT_EQUAL(res, c);
}

UTST_TEST(xyz_norm_square)
{
    XYZ a{1.0, 2.0, 3.0};
    UTST_ASSERT_EQUAL(14.0f, a.norm_square());
}
