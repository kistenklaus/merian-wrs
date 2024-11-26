#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/alias_table/baseline/algo/PartitionAndPrefixSum.hpp"
#include "src/wrs/alias_table/baseline/algo/PrefixSumAvg.hpp"
#include "src/wrs/alias_table/baseline/algo/Split.hpp"
#include "src/wrs/cpu/stable.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

namespace wrs::baseline {

class BaselineAliasTable {
    using weight_t = float;

    static constexpr uint32_t splitCount = 512;

  public:
    BaselineAliasTable(merian::ContextHandle context, uint32_t maxWeights);

    void build(vk::CommandBuffer cmd,
               std::optional<merian::ProfilerHandle> profiler = std::nullopt);

    void set_weights(vk::CommandBuffer cmd,
                     vk::ArrayProxy<weight_t> weights,
                     std::optional<merian::ProfilerHandle> profiler = std::nullopt);

    merian::BufferHandle weightBuffer() const {
        return m_weightBuffer;
    }

    const std::tuple<merian::BufferHandle, uint32_t>
    download_result(vk::CommandBuffer cmd,
                    std::optional<merian::ProfilerHandle> profiler = std::nullopt);

    /* void */
    /* cpuValidation(merian::QueueHandle& queue, merian::CommandPool& cmdPool, uint32_t weightCount); */

  private:
    const merian::ContextHandle m_context;
    const uint32_t m_maxWeights;

    merian::ResourceAllocatorHandle m_alloc;

    merian::BufferHandle m_weightBuffer;
    merian::BufferHandle m_weightBufferStage;

    merian::BufferHandle m_avgPrefixSum;

    merian::BufferHandle m_heavyLight;
    merian::BufferHandle m_heavyLightPrefix;

    merian::BufferHandle m_splitDescriptors;

    merian::BufferHandle m_resultBuffer = nullptr;
    merian::BufferHandle m_resultBufferStage;
    uint32_t m_resultCount = 0;

    baseline::PrefixSumAndAvg m_prefixSumAndAverage;
    baseline::PartitionAndPrefixSum m_partitionAndPrefixSum;
    baseline::Split m_split;
    /* PrefixSum m_prefixSumAlgo; */
};

} // namespace wrs::baseline
