#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/kernels/DecoupledPivotPartitionKernel.hpp"
#include <vulkan/vulkan_handles.hpp>

namespace wrs {

class PivotPartition {
  public:
    PivotPartition(const merian::ContextHandle& context,
                   uint32_t maxElementCount,
                   uint32_t workgroup_size = 512,
                   uint32_t rows = 4)
        : m_partitionKernel(context, workgroup_size, rows) {
        auto resources = context->get_extension<merian::ExtensionResources>();
        assert(resources != nullptr);
        auto alloc = resources->resource_allocator();
        uint32_t partitionSize = m_partitionKernel.partitionSize();
        uint32_t maxWorkgroupCount = (maxElementCount + partitionSize - 1) / partitionSize;
        m_partitionDescriptorBuffer = alloc->createBuffer(
            m_partitionKernel.partitionDescriptorBufferSize(maxWorkgroupCount),
            vk::BufferUsageFlagBits::eStorageBuffer, merian::MemoryMappingType::NONE);
    }

    void run(vk::CommandBuffer cmd,
             merian::BufferHandle in_elements,
             merian::BufferHandle in_pivot,
             merian::BufferHandle out_partition,
             std::optional<uint32_t> elementCount = std::nullopt,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        uint32_t count = elementCount.value_or(in_elements->get_size() / sizeof(float));
        uint32_t partitionSize = m_partitionKernel.partitionSize();
        uint32_t workgroup_count = (count + partitionSize - 1) / partitionSize;

#ifdef MERIAN_PROFILER_ENABLE
        std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
        if (profiler.has_value()) {
            merian_profile_scope.emplace(
                profiler.value(), cmd,
                fmt::format("decoupled pivot partition [workgroupCount = {}]", workgroup_count));
        }
#endif

        m_partitionKernel.dispatch(cmd, workgroup_count, in_elements, count, in_pivot, out_partition,
                                   m_partitionDescriptorBuffer);
    }

  private:
    DecoupledPivotPartitionKernel m_partitionKernel;

    merian::BufferHandle m_partitionDescriptorBuffer;
};
} // namespace wrs
