#include "utst.h"
#include "physics.h"

using namespace CORE;

void test_vel_updated()
{
    VEL v{1.0, 2.0, 3.0};
    ACC a{2.0, 4.0, 6.0};
    VEL v_new = VEL::updated(v, a, 1.0);

    VEL v_new_expected{2.0, 4.0, 6.0};
    UTST_ASSERT_EQUAL(v_new_expected, v_new);
}

int main(int argc, char *argv[])
{
    test_vel_updated();
}