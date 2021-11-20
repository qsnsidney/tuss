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

UTST_TEST(xyz_add)
{
    XYZ a{1.0, 2.0, 3.0};
    XYZ b{11.0, 12.0, 13.0};
    XYZ sum{12.0, 14.0, 16.0};

    XYZ c = a;
    c += b;
    UTST_ASSERT_EQUAL(sum, c);

    XYZ d = a + b;
    UTST_ASSERT_EQUAL(sum, d);
}
