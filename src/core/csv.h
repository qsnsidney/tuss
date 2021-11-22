#pragma once

#include <vector>
#include <istream>
#include "physics.h"

namespace CORE
{
    /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row
    std::vector<BODY_IC> parse_body_ic_from_csv(std::istream &csv_istream);

    std::vector<BODY_IC> parse_body_ic_from_csv(const std::string &csv_file_path);
}