#pragma once

#include "xyz.h"

namespace CORE
{
    using POS = XYZ;
    using VEL = XYZ;
    using ACC = XYZ;

    using POS_VEL_PAIR = std::pair<POS, VEL>;
}
