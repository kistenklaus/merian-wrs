#pragma once

#include "src/wrs/algorithm/split/scalar/test/test_types.hpp"
namespace wrs::test::scalar_split {

static constexpr TestCase TEST_CASES[] = {
  TestCase {
    .weightType = WEIGHT_TYPE_FLOAT,
    .weightCount = 1024 * 2048,
    .distribution = Distribution::PSEUDO_RANDOM_UNIFORM,
    .splitCount = 2048,
    .iterations = 1,
  }
};
}
