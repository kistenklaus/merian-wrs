#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/alias_table/baseline/kernels/DecoupledPartitionAndPrefixSumKernel.hpp"
#include "src/wrs/cpu/stable.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_enums.hpp>
namespace wrs::baseline {

class PartitionAndPrefixSum {
    using weight_t = float;

  public:
    PartitionAndPrefixSum(const merian::ContextHandle& context,
                          uint32_t maxElementCount,
                          uint32_t workgroupSize = 512,
                          uint32_t rows = 5)
        : m_partitionAndPrefixSumKernel(context, workgroupSize, rows) {

        auto resources = context->get_extension<merian::ExtensionResources>();
        assert(resources != nullptr);
        auto alloc = resources->resource_allocator();
        uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
        uint32_t maxWorkgroupCount = (maxElementCount + partitionSize - 1) / partitionSize;
        m_partitionDescriptorBuffer = alloc->createBuffer(
            m_partitionAndPrefixSumKernel.partitionDescriptorBufferSize(maxWorkgroupCount),
            vk::BufferUsageFlagBits::eStorageBuffer, merian::MemoryMappingType::NONE);
    }

    void reset(vk::CommandBuffer cmd, std::optional<uint32_t> elementCount) {
        vk::DeviceSize size;
        if (elementCount.has_value()) {
            uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
            uint32_t workgroupCount = (elementCount.value() + partitionSize - 1) / partitionSize;
            size = m_partitionAndPrefixSumKernel.partitionDescriptorBufferSize(workgroupCount);
        } else {
            size = VK_WHOLE_SIZE;
        }
        cmd.fillBuffer(*m_partitionDescriptorBuffer, 0, size, 0);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {},
            m_partitionDescriptorBuffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                        vk::AccessFlagBits::eShaderRead),
            {});
    }

    void run(vk::CommandBuffer cmd,
             merian::BufferHandle in_elements,
             merian::BufferHandle in_avgPrefixSum,
             merian::BufferHandle out_partitions,
             merian::BufferHandle out_partitionPrefix,
             std::optional<uint32_t> elementCount = std::nullopt,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        uint32_t count = elementCount.value_or(in_elements->get_size() / sizeof(float));
        uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
        uint32_t workgroupCount = (count + partitionSize - 1) / partitionSize;

#ifdef MERIAN_PROFILER_ENABLE
        std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
        if (profiler.has_value()) {
            merian_profile_scope.emplace(
                profiler.value(), cmd,
                fmt::format(
                    "decoupled partition and prefix sum [workgroupCount = {}, optimal={:.3}ms]",
                    workgroupCount, getOptimalTime(count).count()));
        }
#endif

        m_partitionAndPrefixSumKernel.dispatch(cmd, workgroupCount, in_elements, count,
                                               in_avgPrefixSum, out_partitions, out_partitionPrefix,
                                               m_partitionDescriptorBuffer);
    }

    std::chrono::duration<float, std::milli> getOptimalTime(uint32_t elementCount) {
        constexpr uint32_t expectedLookBacks = 4;
        uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
        uint32_t workgroupCount = (elementCount + partitionSize - 1) / partitionSize;
        float requiredTransfer =
            elementCount * (sizeof(weight_t) + sizeof(weight_t) +
                            sizeof(weight_t)) // read weights, write partition and prefix
            + sizeof(weight_t) +              // read pivot
            workgroupCount *
                (sizeof(unsigned int) + 2 * sizeof(weight_t) + 2 * sizeof(unsigned int)) *
                expectedLookBacks;               // lookback accesses
        constexpr float memoryBandwidth = 504e9; // RTX 4070
        return std::chrono::duration<float, std::milli>((requiredTransfer / memoryBandwidth) * 1e3);
    }

  private:
    baseline::DecoupledPartitionAndPrefixSumKernel m_partitionAndPrefixSumKernel;

    merian::BufferHandle m_partitionDescriptorBuffer;
};
} // namespace wrs::baseline
