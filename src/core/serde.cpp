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
    void serialize_system_state_to_csv(std::ostream &csv_ostream, const SYSTEM_STATE &system_state)
    {
        csv_ostream << std::fixed;
        /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row
        for (const auto &[p, v, m] : system_state)
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

    void serialize_system_state_to_csv(const std::string &csv_file_path, const SYSTEM_STATE &system_state)
    {
        std::ofstream csv_file_ofstream(csv_file_path);
        ASSERT(csv_file_ofstream.is_open());
        serialize_system_state_to_csv(csv_file_ofstream, system_state);
    }

    SYSTEM_STATE deserialize_system_state_from_csv(std::istream &csv_istream)
    {
        SYSTEM_STATE system_state;
        /// (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each row
        const std::regex row_regex("(.*),(.*),(.*),(.*),(.*),(.*),(.*),?", std::regex::optimize);

        std::string row_str;

        while (std::getline(csv_istream, row_str))
        {
            std::smatch m;
            std::regex_match(row_str, m, row_regex);
            if (m.size() != 8)
            {
                // Skip invalid row
                std::cout << "Invalid CSV row: " << row_str << std::endl;
                continue;
            }

            POS p{str_to_floating(m[1]), str_to_floating(m[2]), str_to_floating(m[3])};
            VEL v{str_to_floating(m[4]), str_to_floating(m[5]), str_to_floating(m[6])};
            MASS mass{str_to_floating(m[7])};

            system_state.emplace_back(p, v, mass);
        }
        return system_state;
    }

    SYSTEM_STATE deserialize_system_state_from_csv(const std::string &csv_file_path)
    {
        std::ifstream csv_file_ifstream(csv_file_path);
        ASSERT(csv_file_ifstream.is_open());
        return deserialize_system_state_from_csv(csv_file_ifstream);
    }

    void serialize_system_state_to_bin(std::ostream &bin_ostream, const SYSTEM_STATE &system_state)
    {
        // - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
        const int size_floating_value_type = sizeof(UNIVERSE::floating_value_type);
        write_as_binary(bin_ostream, size_floating_value_type);

        /// - second 4 bytes: number of bodies
        const int num_bodies = system_state.size();
        write_as_binary(bin_ostream, num_bodies);

        // - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_STATE
        for (const auto &body_state : system_state)
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

    void serialize_system_state_to_bin(const std::string &bin_file_path, const SYSTEM_STATE &system_state, bool print_file_name)
    {
        std::ofstream bin_file_ofstream(bin_file_path, std::ios::binary);
        if (!bin_file_ofstream.is_open())
        {
            std::cout << "Cannot open " << bin_file_path << std::endl;
            ASSERT(false);
        }

        serialize_system_state_to_bin(bin_file_ofstream, system_state);
        if (print_file_name)
        {
            std::cout << "Successfully wrote to " << bin_file_path << std::endl;
        }
    }

    SYSTEM_STATE deserialize_system_state_from_bin(std::istream &bin_istream)
    {
        SYSTEM_STATE system_state;

        // - first 4 bytes: size of floating type (ie., 4 for floating, 8 for double)
        const auto size_floating_value_type = read_as_binary<int>(bin_istream);
        if (sizeof(UNIVERSE::floating_value_type) != size_floating_value_type)
        {
            std::cout << "Warning: unmatched floating value sizes! Will cast!" << std::endl;
            std::cout << "sizeof(UNIVERSE::floating_value_type)=" << sizeof(UNIVERSE::floating_value_type) << std::endl;
            std::cout << "size_floating_value_type=" << size_floating_value_type << std::endl;
            // ASSERT(sizeof(UNIVERSE::floating_value_type) == size_floating_value_type);
        }

        /// - second 4 bytes: number of bodies
        const auto num_bodies = read_as_binary<int>(bin_istream);
        system_state.reserve(num_bodies);

        // - rest: (POS.x,POS.y,POS.z,VEL.x,VEL.y,VEL.z, MASS) for each BODY_STATE
        for (int count_bodies = 0; count_bodies < num_bodies; count_bodies++)
        {
            POS body_pos;
            VEL body_vel;
            MASS body_mass;

            if (size_floating_value_type == sizeof(double))
            {
                body_pos.x = static_cast<UNIVERSE::floating_value_type>(read_as_binary<double>(bin_istream));
                body_pos.y = static_cast<UNIVERSE::floating_value_type>(read_as_binary<double>(bin_istream));
                body_pos.z = static_cast<UNIVERSE::floating_value_type>(read_as_binary<double>(bin_istream));
                body_vel.x = static_cast<UNIVERSE::floating_value_type>(read_as_binary<double>(bin_istream));
                body_vel.y = static_cast<UNIVERSE::floating_value_type>(read_as_binary<double>(bin_istream));
                body_vel.z = static_cast<UNIVERSE::floating_value_type>(read_as_binary<double>(bin_istream));
                body_mass = static_cast<UNIVERSE::floating_value_type>(read_as_binary<double>(bin_istream));
            }
            else if (size_floating_value_type == sizeof(float))
            {
                body_pos.x = static_cast<UNIVERSE::floating_value_type>(read_as_binary<float>(bin_istream));
                body_pos.y = static_cast<UNIVERSE::floating_value_type>(read_as_binary<float>(bin_istream));
                body_pos.z = static_cast<UNIVERSE::floating_value_type>(read_as_binary<float>(bin_istream));
                body_vel.x = static_cast<UNIVERSE::floating_value_type>(read_as_binary<float>(bin_istream));
                body_vel.y = static_cast<UNIVERSE::floating_value_type>(read_as_binary<float>(bin_istream));
                body_vel.z = static_cast<UNIVERSE::floating_value_type>(read_as_binary<float>(bin_istream));
                body_mass = static_cast<UNIVERSE::floating_value_type>(read_as_binary<float>(bin_istream));
            }
            else
            {
                ASSERT(false && "Unsupported floating value size!");
            }

            system_state.emplace_back(body_pos, body_vel, body_mass);
        }

        return system_state;
    }

    SYSTEM_STATE deserialize_system_state_from_bin(const std::string &bin_file_path)
    {
        std::ifstream bin_file_ifstream(bin_file_path, std::ios::binary);
        ASSERT(bin_file_ifstream.is_open());
        return deserialize_system_state_from_bin(bin_file_ifstream);
    }

    SYSTEM_STATE deserialize_system_state_from_file(const std::string &file_path)
    {
        std::string ext = file_path.substr(file_path.find_last_of(".") + 1);
        if (ext == "csv")
        {
            return deserialize_system_state_from_csv(file_path);
        }
        else if (ext == "bin")
        {
            return deserialize_system_state_from_bin(file_path);
        }
        else
        {
            ASSERT(false && "Unsupported extension");
        }
    }
}