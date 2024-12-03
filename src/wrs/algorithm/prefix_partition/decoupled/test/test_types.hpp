#pragma once

#include "src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartitionKernel.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <stdexcept>
namespace wrs::test::decoupled_prefix_partition {

using Buffers = wrs::DecoupledPrefixPartitionBuffers;

enum WeightT {
    WEIGHT_T_FLOAT,
    /*WEIGHT_T_DOUBLE,*/
    /*WEIGHT_T_UINT,*/
};

static vk::DeviceSize sizeof_weight(WeightT ty) {
    switch (ty) {
    case WEIGHT_T_FLOAT:
        return sizeof(float);
        /*case WEIGHT_T_DOUBLE:*/
        /*    return sizeof(double);*/
        /*case WEIGHT_T_UINT:*/
        /*    return sizeof(uint32_t);*/
    }
    throw std::runtime_error("OH NO");
}

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

    template<typename weight_t> 
    weight_t getPivot() const {
      if constexpr (std::is_same_v<weight_t, float>) {
        return fpivot;
      }else {
        static_assert(false, "Invalid template argument. Invalid weight_t");
      }
    }
};
}
