#pragma once

#include "src/wrs/algorithm/mean/decoupled/DecoupledMean.h"
#include "src/wrs/gen/weight_generator.h"
namespace wrs::test::decoupled_mean {

using Buffers = DecoupledMeanBuffers;

enum ElementType {
    WEIGHT_TYPE_FLOAT,
};

static vk::DeviceSize sizeOfElement(const ElementType wt) {
    switch (wt) {
    case WEIGHT_TYPE_FLOAT:
        return sizeof(float);
    }
}

struct TestCase {
  uint32_t workgroupSize;
  uint32_t rows;
  ElementType elemType;
  uint32_t elementCount;
  Distribution distribution;
  bool stable;
  uint32_t iterations;
};

} // namespace wrs::test::decoupled_mean
