#pragma once

#include "src/wrs/algorithm/pack/scalar/test/test_types.hpp"
namespace wrs::test::scalar_pack {

constexpr TestCase TEST_CASES[] = {
    //
    {
        .weightType = WEIGHT_TYPE_FLOAT,
        .weightCount = 1024,
        .splitCount = 1024 / 32,
        .iterations = 1,
    },
};

}
