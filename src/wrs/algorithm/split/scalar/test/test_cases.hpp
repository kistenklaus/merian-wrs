#pragma once

#include "src/wrs/algorithm/split/scalar/test/test_types.hpp"
namespace wrs::test::scalar_split {

static constexpr TestCase TEST_CASES[] = {
    TestCase{
        .weightType = WEIGHT_TYPE_FLOAT,
        .workgroupSize = 512,
        .weightCount = static_cast<glsl::uint>(1e7),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .splitCount = static_cast<glsl::uint>(1e7) / 8,
        .iterations = 1,
    },
    //
    /* TestCase{ */
    /*     .weightType = WEIGHT_TYPE_FLOAT, */
    /*     .workgroupSize = 512, */
    /*     .weightCount = 256, */
    /*     .distribution = Distribution::SEEDED_RANDOM_UNIFORM, */
    /*     .splitCount = (256) / 8, */
    /*     .iterations = 1, */
    /* }, */
};
}
