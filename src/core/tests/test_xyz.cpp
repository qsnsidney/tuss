#include "utst.h"
#include "xyz.h"

void test_equal()
{
    CORE::XYZ a{1.0, 2.0, 3.0};
    CORE::XYZ b{1.0, 2.0, 3.0};
    UTST_ASSERT_EQUAL(a, a);
    UTST_ASSERT_EQUAL(a, b);
}

void test_add()
{
    CORE::XYZ a{1.0, 2.0, 3.0};
    CORE::XYZ b{11.0, 12.0, 13.0};
    CORE::XYZ sum{12.0, 14.0, 16.0};

    CORE::XYZ c = a;
    c += b;
    UTST_ASSERT_EQUAL(sum, c);

    CORE::XYZ d = a + b;
    UTST_ASSERT_EQUAL(sum, d);
}

int main(int argc, char *argv[])
{
    test_equal();
    test_add();
}