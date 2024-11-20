#pragma once
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_types.hpp"

namespace wrs::test::decoupled_prefix_partition {

constexpr TestCase TEST_CASES[] = {TestCase{
    .workgroupSize = 512,
    .rows = 4,
    .elementCount = 1024 * 2048,
    .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
    .stable = true,
    .writePartition = true,
    .weight_type = WEIGHT_T_FLOAT,
    .fpivot = 0.5f,
}};

}

