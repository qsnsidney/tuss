#include "csv.h"

#include <regex>
#include <fstream>

namespace
{
    template <typename T = CORE::XYZ::value_type>
    T str_to_floating(const std::string &str)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return std::stof(str);
        }
        else
        {
            return std::stod(str);
        }
    }
}

namespace CORE
{
    std::vector<POS_VEL_PAIR> parse_body_ic_from_csv(std::istream &csv_istream)
    {
        std::vector<POS_VEL_PAIR> csv_contents;
        const std::regex row_regex("(.*),(.*),(.*),(.*),(.*),(.*),?", std::regex::optimize);

        std::string row_str;

        while (std::getline(csv_istream, row_str))
        {
            std::cmatch m;
            std::regex_match(row_str, row_regex);
            if (m.size() != 7)
            {
                // Invalid csv
                /// TODO: print some warning messages
                return {};
            }

            POS p{str_to_floating(m[1]), str_to_floating(m[2]), str_to_floating(m[3])};
            VEL v{str_to_floating(m[4]), str_to_floating(m[5]), str_to_floating(m[6])};

            csv_contents.emplace_back(p, v);
        }
        return csv_contents;
    }

    std::vector<POS_VEL_PAIR> parse_body_ic_from_csv(const std::string &csv_file_path)
    {
        std::ifstream csv_file_ifstream(csv_file_path);
        return parse_body_ic_from_csv(csv_file_ifstream);
    }
}