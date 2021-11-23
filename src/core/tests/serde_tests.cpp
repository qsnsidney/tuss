#include "utst.h"
#include "serde.h"
#include "timer.h"

#include <sstream>
#include <iostream>

using namespace CORE;

UTST_MAIN();

UTST_TEST(serialize_deserialize_body_state_vec_to_csv_stream)
{
    BODY_STATE_VEC expected_data{
        {{1.0, -2.0, 3.0}, {4.0, 5.0, -6.0}, 7.0},
        {{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}, 17},
    };

    std::stringstream ss;
    serialize_body_state_vec_to_csv(ss, expected_data);
    BODY_STATE_VEC data = deserialize_body_state_vec_from_csv(ss);

    UTST_ASSERT_EQUAL(expected_data.size(), data.size());
    UTST_ASSERT(expected_data == data);
}

UTST_IGNORED_TEST(bin_to_csv_converter)
{
    TIMER timer("bin_to_csv_converter");

    std::string bin_file = "/Users/lichenliu/p/tuss/benchmark/ic/benchmark_100000.bin";
    BODY_STATE_VEC data = deserialize_body_state_vec_from_bin(bin_file);
    timer.elapsed_previous("deserialize_body_state_vec_from_bin");

    std::string csv_file = "/Users/lichenliu/p/tuss/benchmark/ic/benchmark_100000.csv";
    serialize_body_state_vec_to_csv(csv_file, data);
    timer.elapsed_previous("serialize_body_state_vec_to_csv");

    BODY_STATE_VEC another_data = deserialize_body_state_vec_from_csv(csv_file);
    timer.elapsed_previous("deserialize_body_state_vec_from_csv");

    UTST_ASSERT_EQUAL(data.size(), another_data.size());
    UTST_ASSERT(data == another_data);
}

UTST_IGNORED_TEST(csv_to_bin_converter)
{
    TIMER timer("csv_to_bin_converter");

    std::string csv_file = "/Users/lichenliu/p/tuss/benchmark/ic/benchmark_100000.csv";
    BODY_STATE_VEC data = deserialize_body_state_vec_from_csv(csv_file);
    timer.elapsed_previous("deserialize_body_state_vec_from_csv");

    std::string bin_file = "/Users/lichenliu/p/tuss/benchmark/ic/benchmark_100000.bin";
    serialize_body_state_vec_to_bin(bin_file, data);
    timer.elapsed_previous("serialize_body_state_vec_to_bin");

    BODY_STATE_VEC another_data = deserialize_body_state_vec_from_bin(bin_file);
    timer.elapsed_previous("deserialize_body_state_vec_from_bin");

    UTST_ASSERT_EQUAL(data.size(), another_data.size());
    UTST_ASSERT(data == another_data);
}

UTST_TEST(serialize_deserialize_body_state_vec_to_bin_stream)
{
    BODY_STATE_VEC expected_data{
        {{1.0, -2.0, 3.0}, {4.0, 5.0, -6.0}, 7.0},
        {{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}, 17},
    };

    std::stringstream ss;
    serialize_body_state_vec_to_bin(ss, expected_data);
    BODY_STATE_VEC data = deserialize_body_state_vec_from_bin(ss);

    UTST_ASSERT_EQUAL(expected_data.size(), data.size());
    UTST_ASSERT(expected_data == data);
}