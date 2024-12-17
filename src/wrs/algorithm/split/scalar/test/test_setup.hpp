#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_cases.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_types.hpp"
#include "src/wrs/test/test.hpp"
namespace wrs::test::scalar_split {

inline std::tuple<Buffers, Buffers> allocateBuffers(const wrs::test::TestContext context) {
    uint32_t maxWeightCount = 0;
    uint32_t maxSplitCount = 0;
    for (const auto& testCase : TEST_CASES) {
      maxWeightCount = std::max(maxWeightCount, testCase.weightCount);
      maxSplitCount = std::max(maxSplitCount, testCase.splitCount);
    }
    Buffers buffers = Buffers::allocate(context.alloc,maxWeightCount, maxSplitCount,
        merian::MemoryMappingType::NONE);
    Buffers stage = Buffers::allocate(context.alloc,maxWeightCount, maxSplitCount,
        merian::MemoryMappingType::HOST_ACCESS_RANDOM);
    return std::make_tuple(buffers, stage);
}

} // namespace wrs::test::scalar_split
