#include "utst.h"
#include "serde.h"
#include "timer.h"

#include <sstream>
#include <iostream>

using namespace CORE;

UTST_MAIN();

UTST_TEST(serialize_deserialize_body_ic_to_csv_stream)
{
    std::vector<BODY_IC> expected_data{
        {{{1.0, -2.0, 3.0}}, {{4.0, 5.0, -6.0}}, 7.0},
        {{{11.0, 12.0, 13.0}}, {{14.0, 15.0, 16.0}}, 17},
    };

    std::stringstream ss;
    serialize_body_ic_to_csv(ss, expected_data);
    std::vector<BODY_IC> data = deserialize_body_ic_from_csv(ss);

    UTST_ASSERT_EQUAL(expected_data.size(), data.size());
    UTST_ASSERT(expected_data == data);
}

// UTST_TEST(bin_to_csv_converter)
// {
//     double t0 = CORE::get_time_stamp();

//     std::string bin_file = "/Users/lichenliu/p/TUSS-Tiny-Universe-Simulator-System/benchmark/benchmark_100000.ic.bin";
//     std::vector<BODY_IC> data = deserialize_body_ic_from_bin(bin_file);
//     double t1 = CORE::get_time_stamp();
//     std::cout << "deserialize_body_ic_from_bin=" << t1 - t0 << std::endl;

//     std::string csv_file = "/Users/lichenliu/p/TUSS-Tiny-Universe-Simulator-System/benchmark/benchmark_100000.csv";
//     serialize_body_ic_to_csv(csv_file, data);
//     double t2 = CORE::get_time_stamp();
//     std::cout << "serialize_body_ic_to_csv=" << t2 - t1 << std::endl;

//     std::vector<BODY_IC> another_data = deserialize_body_ic_from_csv(csv_file);
//     double t3 = CORE::get_time_stamp();
//     std::cout << "deserialize_body_ic_from_csv=" << t3 - t2 << std::endl;

//     UTST_ASSERT_EQUAL(data.size(), another_data.size());
//     UTST_ASSERT(data == another_data);
// }

// UTST_TEST(csv_to_bin_converter)
// {
//     double t0 = CORE::get_time_stamp();

//     std::string csv_file = "/Users/lichenliu/p/TUSS-Tiny-Universe-Simulator-System/benchmark/benchmark_100000.csv";
//     std::vector<BODY_IC> data = deserialize_body_ic_from_csv(csv_file);
//     double t1 = CORE::get_time_stamp();
//     std::cout << "deserialize_body_ic_from_csv=" << t1 - t0 << std::endl;

//     std::string bin_file = "/Users/lichenliu/p/TUSS-Tiny-Universe-Simulator-System/benchmark/benchmark_100000.ic.bin";
//     serialize_body_ic_to_bin(bin_file, data);
//     double t2 = CORE::get_time_stamp();
//     std::cout << "serialize_body_ic_to_bin=" << t2 - t1 << std::endl;

//     std::vector<BODY_IC> another_data = deserialize_body_ic_from_bin(bin_file);
//     double t3 = CORE::get_time_stamp();
//     std::cout << "deserialize_body_ic_from_bin=" << t3 - t2 << std::endl;

//     UTST_ASSERT_EQUAL(data.size(), another_data.size());
//     UTST_ASSERT(data == another_data);
// }

UTST_TEST(serialize_deserialize_body_ic_to_bin_stream)
{
    std::vector<BODY_IC> expected_data{
        {{{1.0, -2.0, 3.0}}, {{4.0, 5.0, -6.0}}, 7.0},
        {{{11.0, 12.0, 13.0}}, {{14.0, 15.0, 16.0}}, 17},
    };

    std::stringstream ss;
    serialize_body_ic_to_bin(ss, expected_data);
    std::vector<BODY_IC> data = deserialize_body_ic_from_bin(ss);

    UTST_ASSERT_EQUAL(expected_data.size(), data.size());
    UTST_ASSERT(expected_data == data);
}