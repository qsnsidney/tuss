#include "reference.h"
#include "basic_engine.h"

namespace CPUSIM
{
    bool run_verify_with_reference_engine(CORE::SYSTEM_STATE system_state_ic, const CORE::SYSTEM_STATE &actual_system_state_result, CORE::DT dt, int num_iteration)
    {
        BASIC_ENGINE basic_engine(std::move(system_state_ic), dt, 1, false);
        const CORE::SYSTEM_STATE &reference_system_state_result = basic_engine.run(num_iteration);
        return CORE::verify(reference_system_state_result, actual_system_state_result);
    }
}