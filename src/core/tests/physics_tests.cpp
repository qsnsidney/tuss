#include "utst.h"
#include "physics.h"

using namespace CORE;

UTST_MAIN();

UTST_TEST(vel_updated)
{
    VEL v{{1.0, 2.0, 3.0}};
    ACC a{{2.0, 4.0, 6.0}};
    VEL v_new = VEL::updated(v, a, 2.0);

    VEL v_new_expected{{3.0, 6.0, 9.0}};
    UTST_ASSERT_EQUAL(v_new_expected, v_new);
}

UTST_TEST(pos_updated)
{
    POS p{{10.0, 11.0, 12.0}};
    VEL v{{1.0, 2.0, 3.0}};
    ACC a{{2.0, 4.0, 6.0}};
    POS p_new = POS::updated(p, v, a, 2.0);

    POS p_new_expected{{16.0, 23.0, 30.0}};
    UTST_ASSERT_EQUAL(p_new_expected, p_new);
}