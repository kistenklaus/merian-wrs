#pragma once

#include "src/wrs/algorithm/pack/simd/test/test_types.hpp"
namespace wrs::test::simd_pack {

constexpr TestCase TEST_CASES[] = {
    //
    {
        .weightType = WEIGHT_TYPE_FLOAT,
        .weightCount = static_cast<uint32_t>(1024 * 2048),
        .distribution = Distribution::PSEUDO_RANDOM_UNIFORM,
        .splitCount = static_cast<uint32_t>(1024 * 2048) / 64,
        .workgroupSize = 32,
        .iterations = 16,
    },
};

}
