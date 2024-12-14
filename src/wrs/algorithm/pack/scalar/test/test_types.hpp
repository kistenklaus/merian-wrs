#pragma once

#include "src/wrs/algorithm/pack/scalar/ScalarPack.hpp"
#include "src/wrs/gen/weight_generator.h"
namespace wrs::test::scalar_pack {

using Buffers = wrs::ScalarPackBuffers;

enum WeightType {
    WEIGHT_TYPE_FLOAT,
};

vk::DeviceSize sizeOfWeight(const WeightType ty);

struct TestCase {
    WeightType weightType;
    uint32_t weightCount; // N
    Distribution distribution;
    uint32_t splitCount;  // K
    uint32_t iterations;
};

} // namespace wrs::test::scalar_pack
