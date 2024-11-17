#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/alias_table/baseline/kernels/DecoupledPrefixSumAndAvgKernel.hpp"
namespace wrs::baseline {

class PrefixSumAndAvg {
    using weight_t = float;

  public:
    PrefixSumAndAvg(const merian::ContextHandle& context,
                    uint32_t maxElementCount,
                    uint32_t workgroupSize = 512,
                    uint32_t rows = 5)
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
                fmt::format("decoupled prefix sum and average [workgroupCount = {}, elementCount = {}, optimal = {:.3}ms]", workgroupCount, count, 
                  getOptimalTime(count).count()));
        }
#endif
        m_prefixSumAvg.dispatch(cmd, workgroupCount, count, elements, avgPrefixSum,
                                m_partitionDescriptorBuffer);
    }

    std::chrono::duration<float, std::milli> getOptimalTime(uint32_t elementCount) {
        uint32_t partitionSize = m_prefixSumAvg.partitionSize();
        uint32_t workgroupCount = (elementCount + partitionSize - 1) / partitionSize;
        constexpr uint32_t expectedLookBacks = 4;
        float requiredTransfer = elementCount * (sizeof(weight_t) + sizeof(weight_t)) + sizeof(weight_t) 
              + workgroupCount * (sizeof(unsigned int) + 2 * sizeof(weight_t)) * expectedLookBacks;
        constexpr float memoryBandwidth = 504e9; // RTX 4070
        return std::chrono::duration<float, std::milli>((requiredTransfer / memoryBandwidth) * 1e3);

    }

  private:
    baseline::DecoupledPrefixSumAndAverageKernel m_prefixSumAvg;

    merian::BufferHandle m_partitionDescriptorBuffer;
};
} // namespace wrs
