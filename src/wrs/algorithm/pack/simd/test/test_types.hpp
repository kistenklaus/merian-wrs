#pragma once

#include "src/wrs/algorithm/pack/simd/SimdPack.hpp"
#include "src/wrs/gen/weight_generator.h"

#include <src/wrs/algorithm/pack/simd/SimdPack.hpp>

namespace wrs::test::simd_pack {

using Buffers = SimdPackBuffers;

enum WeightType {
    WEIGHT_TYPE_FLOAT,
};

vk::DeviceSize sizeOfWeight(WeightType ty);

struct TestCase {
    WeightType weightType;
    uint32_t weightCount; // N
    Distribution distribution;
    uint32_t splitCount;  // K
    uint32_t iterations;
};

} // namespace wrs::test::scalar_pack
