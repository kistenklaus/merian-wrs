#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/kernels/PartitionAndPrefixSumKernel.hpp"
namespace wrs {

class PartitionAndPrefixSum {
  public:
    PartitionAndPrefixSum(const merian::ContextHandle& context,
                          uint32_t maxElementCount,
                          uint32_t workgroupSize = 32,
                          uint32_t rows = 1)
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

    void run(vk::CommandBuffer cmd,
             merian::BufferHandle in_elements,
             merian::BufferHandle in_pivot,
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
                fmt::format("decoupled partition and prefix sum [workgroupCount = {}]", workgroupCount));
        }
#endif

        m_partitionAndPrefixSumKernel.dispatch(cmd, workgroupCount, in_elements, count, in_pivot,
                                               out_partitions, out_partitionPrefix,
                                               m_partitionDescriptorBuffer);
    }

  private:
    PartitionAndPrefixSumKernel m_partitionAndPrefixSumKernel;

    merian::BufferHandle m_partitionDescriptorBuffer;
};
} // namespace wrs
