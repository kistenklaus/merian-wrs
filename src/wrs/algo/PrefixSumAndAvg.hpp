#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/kernels/DecoupledPrefixSumAndAvgKernel.hpp"
namespace wrs {

class PrefixSumAndAvg {
    using weight_t = float;

  public:
    PrefixSumAndAvg(const merian::ContextHandle& context,
                    uint32_t maxElementCount,
                    uint32_t workgroupSize = 512,
                    uint32_t rows = 4)
        : m_prefixSumAvg(context, workgroupSize, rows) {
        auto resources = context->get_extension<merian::ExtensionResources>();
        assert(resources != nullptr);
        auto alloc = resources->resource_allocator();
        uint32_t partitionSize = m_prefixSumAvg.partitionSize();
        uint32_t maxWorkgroupCount = (maxElementCount + partitionSize - 1) / partitionSize;
        m_partitionDescriptorBuffer = alloc->createBuffer(
            m_prefixSumAvg.partitionDescriptorBufferSize(maxWorkgroupCount),
            vk::BufferUsageFlagBits::eStorageBuffer, merian::MemoryMappingType::NONE);
    }

    void run(vk::CommandBuffer cmd,
             merian::BufferHandle elements,
             merian::BufferHandle avgPrefixSum,
             std::optional<uint32_t> elementCount = std::nullopt,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        uint32_t count = elementCount.value_or(elements->get_size() / sizeof(weight_t));
        uint32_t partitionSize = m_prefixSumAvg.partitionSize();
        uint32_t workgroupCount = (count + partitionSize - 1) / partitionSize;

#ifdef MERIAN_PROFILER_ENABLE
        std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
        if (profiler.has_value()) {
            merian_profile_scope.emplace(
                profiler.value(), cmd,
                fmt::format("decoupled prefix sum and average [workgroupCount = {}]", workgroupCount));
        }
#endif
        m_prefixSumAvg.dispatch(cmd, workgroupCount, count, elements, avgPrefixSum,
                                m_partitionDescriptorBuffer);
    }

  private:
    DecoupledPrefixSumAndAverageKernel m_prefixSumAvg;

    merian::BufferHandle m_partitionDescriptorBuffer;
};
} // namespace wrs
