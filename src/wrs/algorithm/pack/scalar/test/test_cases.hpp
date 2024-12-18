#pragma once

#include "src/wrs/algorithm/pack/scalar/test/test_types.hpp"
namespace wrs::test::scalar_pack {

constexpr TestCase TEST_CASES[] = {
    //
    {
        .weightType = WEIGHT_TYPE_FLOAT,
        .workgroupSize = 512,
        .weightCount = 1024 * 2048,
        .distribution = Distribution::PSEUDO_RANDOM_UNIFORM,
        .splitCount = 1024 * 2048 / 32,
        .iterations = 1,
    },
};

}
