#pragma once

#include "src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartitionKernel.hpp"
#include "src/wrs/gen/weight_generator.h"
namespace wrs::test::decoupled_prefix_partition {

using Buffers = wrs::DecoupledPrefixPartitionBuffers;

enum WeightT {
    WEIGHT_T_FLOAT,
    /*WEIGHT_T_DOUBLE,*/
    /*WEIGHT_T_UINT,*/
};

vk::DeviceSize sizeOfWeight(WeightT ty);

struct TestCase {
    uint32_t workgroupSize;
    uint32_t rows;
    uint32_t elementCount;
    wrs::Distribution distribution;
    bool stable;
    bool writePartition;
    WeightT weight_type;
    union {
        float fpivot;
    };

    uint32_t iterations;

    template <typename weight_t> weight_t getPivot() const {
        if constexpr (std::is_same_v<weight_t, float>) {
            return fpivot;
        } else {
            static_assert(false, "Invalid template argument. Invalid weight_t");
        }
    }
};
} // namespace wrs::test::decoupled_prefix_partition
