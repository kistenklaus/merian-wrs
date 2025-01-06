#pragma once

#include "src/wrs/algorithm/split/scalar/test/test_types.hpp"
namespace wrs::test::scalar_split {

static constexpr TestCase TEST_CASES[] = {
    TestCase{
        .weightType = WEIGHT_TYPE_FLOAT,
        .workgroupSize = 512,
        .weightCount = 1024 * 2048,
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .splitCount = (1024 * 2048) / 32,
        .iterations = 1,
    },
    //
    TestCase{
        .weightType = WEIGHT_TYPE_FLOAT,
        .workgroupSize = 512,
        .weightCount = 256,
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .splitCount = (256) / 32,
        .iterations = 1,
    },
};
}
