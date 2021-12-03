#pragma once
#include "core/physics.hpp"

namespace CPUSIM
{
    /// Verify with a reference result you can always trust on.
    /// It might be slow, but it will never lie to you.
    bool run_verify_with_reference_engine(CORE::SYSTEM_STATE system_state_ic, const CORE::SYSTEM_STATE &actual_system_state_result, CORE::DT dt, int num_iteration);
}