#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_cases.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_types.hpp"
#include "src/wrs/test/test.hpp"
namespace wrs::test::scalar_split {

inline std::tuple<Buffers, Buffers> allocateBuffers(const wrs::test::TestContext context) {
    vk::DeviceSize maxPartitionPrefixBufferSize = 0;
    vk::DeviceSize maxMeanBufferSize = 0;
    vk::DeviceSize maxSplitBufferSize = 0;

    for (const auto& testCase : TEST_CASES) {
        vk::DeviceSize partitionBufferSize = Buffers::minPartitionPrefixBufferSize(
            testCase.weightCount, sizeOfWeightType(testCase.weightType));
        vk::DeviceSize splitBufferSize =
            Buffers::minSplitBufferSize(testCase.splitCount, sizeOfWeightType(testCase.weightType));
        vk::DeviceSize meanBufferSize =
            Buffers::minMeanBufferSize(sizeOfWeightType(testCase.weightType));

        maxPartitionPrefixBufferSize = std::max(maxPartitionPrefixBufferSize, partitionBufferSize);
        maxSplitBufferSize = std::max(maxSplitBufferSize, splitBufferSize);
        maxMeanBufferSize = std::max(maxMeanBufferSize, meanBufferSize);
    }
    maxMeanBufferSize *= 10;
    maxSplitBufferSize *= 10;
    maxPartitionPrefixBufferSize *= 10;

    Buffers buffers;
    Buffers stage;
    buffers.partitionPrefix = context.alloc->createBuffer(
        maxPartitionPrefixBufferSize,
        Buffers::PARTITION_PREFIX_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst,
        merian::MemoryMappingType::NONE);

    stage.partitionPrefix = context.alloc->createBuffer(
        maxPartitionPrefixBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
        merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.mean = context.alloc->createBuffer(
        maxMeanBufferSize, Buffers::MEAN_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst,
        merian::MemoryMappingType::NONE);
    stage.mean =
        context.alloc->createBuffer(maxMeanBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.splits = context.alloc->createBuffer(maxSplitBufferSize,
                                                 Buffers::SPLITS_BUFFER_USAGE_FLAGS |
                                                     vk::BufferUsageFlagBits::eTransferSrc,
                                                 merian::MemoryMappingType::NONE);
    stage.splits =
        context.alloc->createBuffer(maxSplitBufferSize, vk::BufferUsageFlagBits::eTransferDst,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    return std::make_tuple(buffers, stage);
}

} // namespace wrs::test::scalar_split
