#pragma once

#include "src/wrs/algorithm/pack/scalar/test/test_types.hpp"
namespace wrs::test::scalar_pack {

constexpr TestCase TEST_CASES[] = {
    //
    {
        .weightType = WEIGHT_TYPE_FLOAT,
        .workgroupSize = 512,
        .weightCount = static_cast<glsl::uint>(1e7),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .splitCount = static_cast<glsl::uint>(1e7) / 8,
        .iterations = 1,
    },
    /*{*/
    /*    .weightType = WEIGHT_TYPE_FLOAT,*/
    /*    .workgroupSize = 512,*/
    /*    .weightCount = 256,*/
    /*    .distribution = Distribution::SEEDED_RANDOM_UNIFORM,*/
    /*    .splitCount = 256 / 8,*/
    /*    .iterations = 1,*/
    /*},*/
};

}
