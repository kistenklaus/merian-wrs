#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_cases.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_types.hpp"
#include "src/wrs/test/test.hpp"
#include <vulkan/vulkan_core.h>

namespace wrs::test::decoupled_prefix_partition {

inline std::tuple<Buffers, Buffers> allocateBuffers(const wrs::test::TestContext& context) {
    uint32_t maxElementCount = 0;
    std::size_t maxPartitionSize = 0;
    for (const auto& testCase : TEST_CASES) {
      maxElementCount = std::max(maxElementCount, testCase.elementCount);
      maxPartitionSize = std::max(maxPartitionSize, Buffers::partitionSize(testCase.workgroupSize, testCase.rows));
    }

    Buffers buffers = Buffers::allocate(context.alloc, maxElementCount, maxPartitionSize, merian::MemoryMappingType::NONE);
    Buffers stage = Buffers::allocate(context.alloc, maxElementCount, maxPartitionSize, merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    return std::make_tuple(buffers, stage);
}

} // namespace wrs::test::decoupled_prefix_partition
