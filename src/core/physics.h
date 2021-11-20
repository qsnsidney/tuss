#pragma once

#include "xyz.h"
#include "universe.h"

namespace CORE
{
    using DT = UNIVERSE::floating_value_type;

    struct ACC : public XYZ
    {
    };

    struct VEL : public XYZ
    {
        static VEL updated(VEL, ACC, DT);
    };
    struct POS : public XYZ
    {
        static POS updated(POS, VEL, ACC, DT);
    };

    using POS_VEL_PAIR = std::pair<POS, VEL>;
}
