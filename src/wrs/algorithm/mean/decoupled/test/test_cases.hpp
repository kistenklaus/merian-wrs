#pragma once
#include "./test_types.hpp"

namespace wrs::test::decoupled_mean {

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .workgroupSize = 512,
        .rows = 8,
        .elemType = WEIGHT_TYPE_FLOAT,
        .elementCount = static_cast<uint32_t>(1024 * 2048),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = false,
        .iterations = 1,
    },

    TestCase{
        .workgroupSize = 64,
        .rows = 2,
        .elemType = WEIGHT_TYPE_FLOAT,
        .elementCount = 256,
        .distribution = Distribution::UNIFORM,
        .stable = false,
        .iterations = 1,
    },
};

}
