#pragma once
#include "./test_types.hpp"

namespace wrs::test::decoupled_mean {

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .workgroupSize = 512,
        .rows = 4,
        .elemType = WEIGHT_TYPE_FLOAT,
        .elementCount = static_cast<uint32_t>(1024 * 2048),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = false,
        .iterations = 1,
    }};

}
