#include "utst.h"
#include "csv.h"

#include <sstream>

using namespace CORE;

UTST_MAIN();

UTST_TEST(parse_body_ic_from_csv_istream)
{
    std::stringstream ss;
    ss << "1,-2,+3.0,4e0,5.e0, -6.0e0, 7" << std::endl;
    ss << "11,12,13,14,15,16,17" << std::endl;

    std::vector<BODY_IC> data = parse_body_ic_from_csv(ss);

    std::vector<BODY_IC> expected_data{
        {{1.0, -2.0, 3.0}, {4.0, 5.0, -6.0}, 7.0},
        {{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}, 17},
    };

    UTST_ASSERT(expected_data == data);
}