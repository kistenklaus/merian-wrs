#pragma once

#include "./test_types.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/algorithm/mean/decoupled/test/test_cases.hpp"
#include "src/wrs/test/test.hpp"
#include <fmt/base.h>
#include <tuple>
#include <vulkan/vulkan_enums.hpp>

namespace wrs::test::decoupled_mean {

inline std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {

    uint32_t maxElementCount;
    std::size_t maxPartitionSize;

    for (const auto& testCase : TEST_CASES) {
      maxElementCount = std::max(maxElementCount, testCase.elementCount);
      maxPartitionSize = std::max(maxPartitionSize, Buffers::partitionSize(testCase.workgroupSize, testCase.rows));
    }

    Buffers buffers = Buffers::allocate(context.alloc, maxElementCount, maxPartitionSize, 
        merian::MemoryMappingType::NONE);

    Buffers stage = Buffers::allocate(context.alloc, maxElementCount, maxPartitionSize, 
        merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    return std::make_tuple(buffers, stage);
}

} // namespace wrs::test::decoupled_mean
