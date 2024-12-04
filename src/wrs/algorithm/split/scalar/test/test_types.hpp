#pragma once

#include "src/wrs/algorithm/split/scalar/ScalarSplit.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <stdexcept>
namespace wrs::test::scalar_split {

using Buffers = ScalarSplitBuffers;

enum WeightType {
    WEIGHT_TYPE_FLOAT,
};

static constexpr vk::DeviceSize sizeOfWeightType(WeightType type) {
    switch (type) {
    case WEIGHT_TYPE_FLOAT:
        return sizeof(float);
    }
    throw std::runtime_error("sizeOfWeightType is not implemented properly");
}

struct TestCase {
    WeightType weightType;
    uint32_t weightCount;
    Distribution distribution;
    uint32_t splitCount;
    uint32_t iterations;
};

} // namespace wrs::test::scalar_split
