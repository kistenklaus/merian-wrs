#pragma once

#include "./test_types.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/algorithm/mean/decoupled/test/test_cases.hpp"
#include "src/wrs/test/test.hpp"
#include <fmt/base.h>
#include <tuple>
#include <vulkan/vulkan_enums.hpp>

namespace wrs::test::decoupled_mean {

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    Buffers buffers;
    Buffers stage;

    vk::DeviceSize maxElementBufferSize = 0;
    vk::DeviceSize maxMeanBufferSize = 0;
    vk::DeviceSize maxDecoupledStateBufferSize = 0;
    vk::DeviceSize maxDecoupledAggBufferSize = 0;

    for (const auto& testCase : TEST_CASES) {
        vk::DeviceSize elementBufferSize = testCase.elementCount * sizeOfElement(testCase.elemType);
        vk::DeviceSize meanBufferSize = sizeOfElement(testCase.elemType);
        vk::DeviceSize decoupledStateBufferSize = Buffers::minDecoupledStateSize(
            testCase.elementCount, testCase.workgroupSize, testCase.rows);

        fmt::println("BUFFER-SIZE : {}bytes", decoupledStateBufferSize);

        vk::DeviceSize decoupledAggBufferSize =
            Buffers::minDecoupledAggregatesSize(testCase.elementCount, testCase.workgroupSize,
                                                testCase.rows, sizeOfElement(testCase.elemType));

        maxElementBufferSize = std::max(maxElementBufferSize, elementBufferSize);
        maxMeanBufferSize = std::max(maxMeanBufferSize, meanBufferSize);
        maxDecoupledStateBufferSize =
            std::max(maxDecoupledStateBufferSize, decoupledStateBufferSize);
        maxDecoupledAggBufferSize = std::max(maxDecoupledAggBufferSize, decoupledAggBufferSize);
    }

    maxElementBufferSize *= 2;
    maxMeanBufferSize *= 2;
    maxDecoupledStateBufferSize *= 4;
    maxDecoupledAggBufferSize *= 2;

    buffers.elements = context.alloc->createBuffer(maxElementBufferSize,
                                                   Buffers::ELEMENT_BUFFER_USAGE_FLAGS |
                                                       vk::BufferUsageFlagBits::eTransferDst,
                                                   merian::MemoryMappingType::NONE);
    stage.elements =
        context.alloc->createBuffer(maxElementBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.mean = context.alloc->createBuffer(
        maxMeanBufferSize, Buffers::MEAN_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferSrc,
        merian::MemoryMappingType::NONE);
    stage.mean =
        context.alloc->createBuffer(maxMeanBufferSize, vk::BufferUsageFlagBits::eTransferDst,
                                    merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.decoupledStates = context.alloc->createBuffer(maxDecoupledStateBufferSize,
                                                          Buffers::DECOUPLED_STATE_USAGE_FLAGS |
                                                              vk::BufferUsageFlagBits::eTransferSrc,
                                                          merian::MemoryMappingType::NONE);

    stage.decoupledStates = context.alloc->createBuffer(
        maxDecoupledStateBufferSize,
        Buffers::DECOUPLED_STATE_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst,
        merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    buffers.decoupledAggregates = context.alloc->createBuffer(
        maxDecoupledAggBufferSize, Buffers::DECOUPLED_AGGREGATES_BUFFER_USAGE_FLAGS 
        | vk::BufferUsageFlagBits::eTransferSrc,
        merian::MemoryMappingType::NONE);
    stage.decoupledAggregates = context.alloc->createBuffer(
        maxDecoupledAggBufferSize,
        Buffers::DECOUPLED_AGGREGATES_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst,
        merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    return std::make_tuple(buffers, stage);
}

} // namespace wrs::test::decoupled_mean
