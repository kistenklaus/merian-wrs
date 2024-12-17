#pragma once

#include "src/wrs/algorithm/pack/scalar/test/test_types.hpp"
#include "src/wrs/test/test.hpp"
#include <tuple>
namespace wrs::test::simd_pack {


inline std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
  std::uint32_t maxWeightCount = 0;
  std::uint32_t maxSplitCount = 0;
  for (const auto& testCase : TEST_CASES) {
    maxWeightCount = std::max(maxWeightCount, testCase.weightCount);
    maxSplitCount = std::max(maxSplitCount, testCase.splitCount);
  }
  Buffers buffers = Buffers::allocate(context.alloc, maxWeightCount, maxSplitCount, 
      merian::MemoryMappingType::NONE);  
  Buffers stage = Buffers::allocate(context.alloc, maxWeightCount, maxSplitCount, 
      merian::MemoryMappingType::HOST_ACCESS_RANDOM);  

  return std::make_tuple(buffers, stage);
}

}
