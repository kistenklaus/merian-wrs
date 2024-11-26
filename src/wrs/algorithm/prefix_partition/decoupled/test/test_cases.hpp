#pragma once
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_types.hpp"

namespace wrs::test::decoupled_prefix_partition {

constexpr TestCase TEST_CASES[] = {
    TestCase{
        .workgroupSize = 512,
        .rows = 5,
        .elementCount = 1024 * 2048,
        .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = true,
        .writePartition = true,
        .weight_type = WEIGHT_T_FLOAT,
        .fpivot = 0.5f,
        .iterations = 4,
    },
    TestCase{
        .workgroupSize = 512,
        .rows = 5,
        .elementCount = 1024 * 2048,
        .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = true,
        .writePartition = false,
        .weight_type = WEIGHT_T_FLOAT,
        .fpivot = 0.5f,
        .iterations = 4,
    },
    TestCase{
        .workgroupSize = 512,
        .rows = 5,
        .elementCount = 1024 * 2048,
        .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = false,
        .writePartition = true,
        .weight_type = WEIGHT_T_FLOAT,
        .fpivot = 0.5f,
        .iterations = 4,
    },
    TestCase{
        .workgroupSize = 512,
        .rows = 5,
        .elementCount = 1024 * 2048,
        .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = false,
        .writePartition = false,
        .weight_type = WEIGHT_T_FLOAT,
        .fpivot = 0.5f,
        .iterations = 4,
    },
};

}
