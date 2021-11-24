#include "serde.h"
#include "macros.hpp"

#include <regex>
#include <fstream>
#include <iostream>

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

    template <typename T>
    void write_as_binary(std::ostream &os, T value)
    {
        os.write(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    template <typename T>
    T read_as_binary(std::istream &is)
    {
        T value;
        is.read(reinterpret_cast<char *>(&value), sizeof(T));
        return value;
    }
}

namespace CORE
{
    void serialize_body_state_vec_to_csv(std::ostream &csv_ostream, const BODY_STATE_VEC &body_states)
    {
        csv_ostream << std::fixed;
        /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row
        for (const auto &[p, v, m] : body_states)
        {
            csv_ostream << p.x << ",";
            csv_ostream << p.y << ",";
            csv_ostream << p.z << ",";
            csv_ostream << v.x << ",";
            csv_ostream << v.y << ",";
            csv_ostream << v.z << ",";
            csv_ostream << m << "\n";
        }
    }

    void serialize_body_state_vec_to_csv(const std::string &csv_file_path, const BODY_STATE_VEC &body_states)
    {
        std::ofstream csv_file_ofstream(csv_file_path);
        ASSERT(csv_file_ofstream.is_open());
        serialize_body_state_vec_to_csv(csv_file_ofstream, body_states);
    }

    BODY_STATE_VEC deserialize_body_state_vec_from_csv(std::istream &csv_istream)
    {
        BODY_STATE_VEC body_states;
        /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row
        const std::regex row_regex("(.*),(.*),(.*),(.*),(.*),(.*),(.*),?", std::regex::optimize);

        std::string row_str;

        while (std::getline(csv_istream, row_str))
        {
            std::smatch m;
            std::regex_match(row_str, m, row_regex);
            if (m.size() != 8)
            {
                // Invalid csv
                std::cout << "Invalid CSV: " << m.size() << " matches for row:" << std::endl;
                std::cout << row_str << std::endl;
                return {};
            }

            POS p{str_to_floating(m[1]), str_to_floating(m[2]), str_to_floating(m[3])};
            VEL v{str_to_floating(m[4]), str_to_floating(m[5]), str_to_floating(m[6])};
            MASS mass{str_to_floating(m[7])};

            body_states.emplace_back(p, v, mass);
        }
        return body_states;
    }

    BODY_STATE_VEC deserialize_body_state_vec_from_csv(const std::string &csv_file_path)
    {
        std::ifstream csv_file_ifstream(csv_file_path);
        ASSERT(csv_file_ifstream.is_open());
        return deserialize_body_state_vec_from_csv(csv_file_ifstream);
    }

    void serialize_body_state_vec_to_bin(std::ostream &bin_ostream, const BODY_STATE_VEC &body_states)
    {
        // - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
        const int size_floating_value_type = sizeof(UNIVERSE::floating_value_type);
        write_as_binary(bin_ostream, size_floating_value_type);

        /// - second 4 bytes: number of bodies
        const int num_bodies = body_states.size();
        write_as_binary(bin_ostream, num_bodies);

        // - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_STATE
        for (const auto &body_state : body_states)
        {
            const auto &body_pos = std::get<POS>(body_state);
            const auto &body_vel = std::get<VEL>(body_state);
            const auto body_mass = std::get<MASS>(body_state);

            write_as_binary(bin_ostream, body_pos.x);
            write_as_binary(bin_ostream, body_pos.y);
            write_as_binary(bin_ostream, body_pos.z);
            write_as_binary(bin_ostream, body_vel.x);
            write_as_binary(bin_ostream, body_vel.y);
            write_as_binary(bin_ostream, body_vel.z);
            write_as_binary(bin_ostream, body_mass);
        }
    }

    void serialize_body_state_vec_to_bin(const std::string &bin_file_path, const BODY_STATE_VEC &body_states)
    {
        std::ofstream bin_file_ofstream(bin_file_path, std::ios::binary);
        if (!bin_file_ofstream.is_open())
        {
            std::cout << "Cannot open " << bin_file_path << std::endl;
            ASSERT(false);
        }

        serialize_body_state_vec_to_bin(bin_file_ofstream, body_states);
    }

    BODY_STATE_VEC deserialize_body_state_vec_from_bin(std::istream &bin_istream)
    {
        BODY_STATE_VEC body_states;

        // - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
        const int expected_size_floating_value_type = sizeof(UNIVERSE::floating_value_type);
        const auto size_floating_value_type = read_as_binary<int>(bin_istream);
        ASSERT(expected_size_floating_value_type == size_floating_value_type);

        /// - second 4 bytes: number of bodies
        const auto num_bodies = read_as_binary<int>(bin_istream);
        body_states.reserve(num_bodies);

        // - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_STATE
        for (int count_bodies = 0; count_bodies < num_bodies; count_bodies++)
        {
            POS body_pos;
            VEL body_vel;
            MASS body_mass;

            body_pos.x = read_as_binary<UNIVERSE::floating_value_type>(bin_istream);
            body_pos.y = read_as_binary<UNIVERSE::floating_value_type>(bin_istream);
            body_pos.z = read_as_binary<UNIVERSE::floating_value_type>(bin_istream);
            body_vel.x = read_as_binary<UNIVERSE::floating_value_type>(bin_istream);
            body_vel.y = read_as_binary<UNIVERSE::floating_value_type>(bin_istream);
            body_vel.z = read_as_binary<UNIVERSE::floating_value_type>(bin_istream);
            body_mass = read_as_binary<UNIVERSE::floating_value_type>(bin_istream);

            body_states.emplace_back(body_pos, body_vel, body_mass);
        }

        return body_states;
    }

    BODY_STATE_VEC deserialize_body_state_vec_from_bin(const std::string &bin_file_path)
    {
        std::ifstream bin_file_ifstream(bin_file_path, std::ios::binary);
        ASSERT(bin_file_ifstream.is_open());
        return deserialize_body_state_vec_from_bin(bin_file_ifstream);
    }

    BODY_STATE_VEC deserialize_body_state_vec_from_file(const std::string &file_path)
    {
        std::string ext = file_path.substr(file_path.find_last_of(".") + 1);
        if (ext == "csv")
        {
            return deserialize_body_state_vec_from_csv(file_path);
        }
        else if (ext == "bin")
        {
            return deserialize_body_state_vec_from_bin(file_path);
        }
        else
        {
            ASSERT(false && "Unsupported extension");
        }
    }
}