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
    Buffers buffers;
    Buffers stage;

    vk::DeviceSize maxElementBufferSize = 0;
    vk::DeviceSize maxMeanBufferSize = 0;
    vk::DeviceSize maxDecoupledStateBufferSize = 0;

    for (const auto& testCase : TEST_CASES) {
        vk::DeviceSize elementSize = sizeOfElement(testCase.elemType);

        vk::DeviceSize elementBufferSize =
            Buffers::minElementBufferSize(testCase.elementCount, elementSize);
        vk::DeviceSize meanBufferSize = Buffers::minMeanBufferSize(elementSize);
        vk::DeviceSize decoupledStateBufferSize = Buffers::minDecoupledStateSize(
            testCase.elementCount, testCase.workgroupSize, testCase.rows);

        fmt::println("BUFFER-SIZE : {}bytes", decoupledStateBufferSize);

        maxElementBufferSize = std::max(maxElementBufferSize, elementBufferSize);
        maxMeanBufferSize = std::max(maxMeanBufferSize, meanBufferSize);
        maxDecoupledStateBufferSize =
            std::max(maxDecoupledStateBufferSize, decoupledStateBufferSize);
    }

    maxElementBufferSize *= 2;
    maxMeanBufferSize *= 2;
    maxDecoupledStateBufferSize *= 4;

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

    return std::make_tuple(buffers, stage);
}

} // namespace wrs::test::decoupled_mean
