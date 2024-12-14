#pragma once

#include "src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartitionKernel.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_cases.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_types.hpp"
#include "src/wrs/test/test.hpp"
#include <vulkan/vulkan_core.h>

namespace wrs::test::decoupled_prefix_partition {

inline std::tuple<Buffers, Buffers> allocateBuffers(const wrs::test::TestContext& context) {
    vk::DeviceSize maxElementBufferSize = 0;
    vk::DeviceSize maxPivotBufferSize = 0;
    vk::DeviceSize maxBatchBufferSize = 0;
    vk::DeviceSize maxPrefixBufferSize = 0;
    vk::DeviceSize maxPartitionBufferSize = 0;
    for (const auto& testCase : TEST_CASES) {
        vk::DeviceSize elementBufferSize =
            sizeOfWeight(testCase.weight_type) * testCase.elementCount;
        vk::DeviceSize pivotBufferSize = sizeOfWeight(testCase.weight_type);
        vk::DeviceSize batchBufferSize =
            wrs::DecoupledPrefixPartitionBuffers::minBatchDescriptorSize(
                testCase.elementCount, testCase.workgroupSize * testCase.rows,
                sizeOfWeight(testCase.weight_type));
        vk::DeviceSize prefixBufferSize =
            sizeof(uint32_t) + sizeOfWeight(testCase.weight_type) * testCase.elementCount;

        vk::DeviceSize partitionBufferSize =
            testCase.writePartition ? sizeOfWeight(testCase.weight_type) * testCase.elementCount
                                    : 0;

        maxElementBufferSize = std::max(elementBufferSize, maxElementBufferSize);
        maxPivotBufferSize = std::max(pivotBufferSize, maxPivotBufferSize);
        maxBatchBufferSize = std::max(batchBufferSize, maxBatchBufferSize);
        maxPrefixBufferSize = std::max(prefixBufferSize, maxPrefixBufferSize);
        maxPartitionBufferSize = std::max(partitionBufferSize, maxPartitionBufferSize);
    }

    Buffers buffers;
    Buffers stage;

    buffers.elements = context.alloc->createBuffer(maxElementBufferSize,
                                                   Buffers::ELEMENT_BUFFER_USAGE_FLAGS |
                                                       vk::BufferUsageFlagBits::eTransferDst,
                                                   merian::MemoryMappingType::NONE);
    stage.elements =
        context.alloc->createBuffer(maxElementBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.pivot = context.alloc->createBuffer(maxPivotBufferSize,
                                                Buffers::PIVOT_BUFFER_USAGE_FLAGS |
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                merian::MemoryMappingType::NONE);
    stage.pivot =
        context.alloc->createBuffer(maxPivotBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.batchDescriptors = context.alloc->createBuffer(
        maxBatchBufferSize,
        Buffers::BATCH_DESCRIPTOR_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst,
        merian::MemoryMappingType::NONE);
    stage.batchDescriptors = context.alloc->createBuffer(
        maxBatchBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.partitionPrefix = context.alloc->createBuffer(maxPrefixBufferSize,
                                                          Buffers::PREFIX_BUFFER_USAGE_FLAGS |
                                                              vk::BufferUsageFlagBits::eTransferSrc,
                                                          merian::MemoryMappingType::NONE);
    stage.partitionPrefix =
        context.alloc->createBuffer(maxPrefixBufferSize, vk::BufferUsageFlagBits::eTransferDst,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    if (maxPartitionBufferSize > 0) {
        buffers.partition = context.alloc->createBuffer(maxPartitionBufferSize,
                                                        Buffers::PARTITION_BUFFER_USAGE_FLAGS |
                                                            vk::BufferUsageFlagBits::eTransferSrc,
                                                        merian::MemoryMappingType::NONE);
        stage.partition = context.alloc->createBuffer(
            maxPartitionBufferSize, vk::BufferUsageFlagBits::eTransferDst,
            merian::MemoryMappingType::HOST_ACCESS_RANDOM);
    }else {
      buffers.partition = VK_NULL_HANDLE;
      stage.partition = VK_NULL_HANDLE;
    }

    return std::make_tuple(buffers, stage);
}

} // namespace wrs::test::decoupled_prefix_partition
