#pragma once

#include <vector>
#include <istream>
#include <string_view>
#include "physics.h"

namespace CORE
{
    /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z) for each row
    std::vector<POS_VEL_PAIR> parse_body_ic_from_csv(std::istream &csv_istream);

    std::vector<POS_VEL_PAIR> parse_body_ic_from_csv(std::string_view csv_file_path);
}