#pragma once

#include "src/wrs/algorithm/split/scalar/ScalarSplit.hpp"
#include "src/wrs/gen/weight_generator.h"
namespace wrs::test::scalar_split {

using Buffers = ScalarSplitBuffers;

enum WeightType {
    WEIGHT_TYPE_FLOAT,
};

vk::DeviceSize sizeOfWeightType(WeightType type);

struct TestCase {
    WeightType weightType;
    uint32_t weightCount;
    Distribution distribution;
    uint32_t splitCount;
    uint32_t iterations;
};

} // namespace wrs::test::scalar_split
