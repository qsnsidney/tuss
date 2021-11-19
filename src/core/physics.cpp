#include "physics.h"

namespace CORE
{
    VEL VEL::updated(VEL v, ACC a, DT dt)
    {
        return {v + static_cast<XYZ::value_type>(0.5) * a * dt};
    }
}