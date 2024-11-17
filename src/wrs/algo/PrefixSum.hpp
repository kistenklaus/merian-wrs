#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/kernels/DecoupledPrefixSumKernel.hpp"

namespace wrs {

class PrefixSum {
    using weight_t = DecoupledPrefixSumKernel::weight_t;

  public:
    PrefixSum(const merian::ContextHandle& context, uint32_t maxElementCount)
        : m_prefixSumKernel(context, 512,4) {
        auto resources = context->get_extension<merian::ExtensionResources>();
        assert(resources != nullptr);
        auto alloc = resources->resource_allocator();
        uint32_t partitionSize = m_prefixSumKernel.partitionSize();
        uint32_t maxWorkgroupCount = (maxElementCount + partitionSize - 1) / partitionSize;
        m_partitionDescriptorBuffer = alloc->createBuffer(
            m_prefixSumKernel.partitionDescriptorBufferSize(maxWorkgroupCount),
            vk::BufferUsageFlagBits::eStorageBuffer, merian::MemoryMappingType::NONE);
    }

    void run(vk::CommandBuffer cmd,
             merian::BufferHandle in_elements,
             merian::BufferHandle out_prefixSum,
             std::optional<uint32_t> elementCount = std::nullopt, 
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        uint32_t count = elementCount.value_or(in_elements->get_size() / sizeof(weight_t));
        uint32_t partitionSize = m_prefixSumKernel.partitionSize();
        uint32_t workgroupCount = (count + partitionSize - 1) / partitionSize;

#ifdef MERIAN_PROFILER_ENABLE
        std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
        if (profiler.has_value()) {
            merian_profile_scope.emplace(profiler.value(), cmd, fmt::format("prefix sum [workgroupCount = {}]", workgroupCount));
        }
#endif

        m_prefixSumKernel.dispatch(cmd, workgroupCount, in_elements, out_prefixSum, m_partitionDescriptorBuffer);
    }

  private:
    DecoupledPrefixSumKernel m_prefixSumKernel;
    merian::BufferHandle m_partitionDescriptorBuffer;
};

} // namespace wrs
