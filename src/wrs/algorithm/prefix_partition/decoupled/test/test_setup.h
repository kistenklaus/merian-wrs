#pragma once

#include "src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartitionKernel.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_cases.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_types.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/test/test.hpp"
#include <memory>
#include <memory_resource>
#include <type_traits>

namespace wrs::test::decoupled_prefix_partition {

static std::tuple<Buffers, Buffers, std::unique_ptr<std::pmr::memory_resource>>
allocateBuffers(const wrs::test::TestContext& context) {
    vk::DeviceSize maxElementBufferSize = 0;
    vk::DeviceSize maxPivotBufferSize = 0;
    vk::DeviceSize maxBatchBufferSize = 0;
    vk::DeviceSize maxPrefixBufferSize = 0;
    vk::DeviceSize maxPartitionBufferSize = 0;
    for (const auto& testCase : TEST_CASES) {
        vk::DeviceSize elementBufferSize =
            sizeof_weight(testCase.weight_type) * testCase.elementCount;
        vk::DeviceSize pivotBufferSize = sizeof_weight(testCase.weight_type);
        vk::DeviceSize batchBufferSize =
            wrs::DecoupledPrefixPartitionKernelBuffers::minBatchDescriptorSize(
                testCase.elementCount, testCase.workgroupSize * testCase.rows,
                sizeof_weight(testCase.weight_type));
        vk::DeviceSize prefixBufferSize =
            sizeof(uint32_t) + sizeof_weight(testCase.weight_type) * testCase.elementCount;

        vk::DeviceSize partitionBufferSize =
            testCase.writePartition ? sizeof_weight(testCase.weight_type) * testCase.elementCount
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
                                    merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);

    buffers.pivot = context.alloc->createBuffer(maxPivotBufferSize,
                                                Buffers::PIVOT_BUFFER_USAGE_FLAGS |
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                merian::MemoryMappingType::NONE);
    stage.pivot =
        context.alloc->createBuffer(maxPivotBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                    merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);

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

    buffers.partition = context.alloc->createBuffer(maxPartitionBufferSize,
                                                    Buffers::PARTITION_BUFFER_USAGE_FLAGS |
                                                        vk::BufferUsageFlagBits::eTransferSrc,
                                                    merian::MemoryMappingType::NONE);
    stage.partition =
        context.alloc->createBuffer(maxPartitionBufferSize, vk::BufferUsageFlagBits::eTransferDst,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    std::unique_ptr<std::pmr::memory_resource> resource =
        std::make_unique<std::pmr::monotonic_buffer_resource>(maxElementBufferSize * 10);

    return std::make_tuple(buffers, stage, std::move(resource));
}

} // namespace wrs::test::decoupled_prefix_partition
