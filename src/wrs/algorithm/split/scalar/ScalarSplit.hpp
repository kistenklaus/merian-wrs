#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include "src/wrs/types/split.hpp"
#include "src/wrs/why.hpp"
#include <concepts>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct ScalarSplitBuffers {
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    using weight_type = glsl::float_t;

    /**
     * Buffer which contains the prefix sums over both the heavy and light
     * partitions.
     *
     * Layout:
     * First 4byte word contains the amount of heavy items
     * The following heavyCount 4byte words contain the prefix sum over
     * heavy items in accending order.
     * After the prefix sum over the heavy items
     * the prefix sum of the light elements in decending order.
     *
     */
    merian::BufferHandle partitionPrefix;
    static constexpr vk::BufferUsageFlags PARTITION_PREFIX_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    static constexpr vk::DeviceSize minPartitionPrefixBufferSize(uint32_t N, size_t sizeOfWeight) {
        return sizeof(uint32_t) + sizeof(uint32_t) + sizeOfWeight * N;
    }
    using PartitionPrefixLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<wrs::glsl::uint, layout::StaticString("heavyCount")>,
                             layout::Attribute<weight_type*, layout::StaticString("heavyLightIndices")>>;
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout>;

    /**
     * Buffer which simply contains the average weight.
     */
    merian::BufferHandle mean;
    static constexpr vk::BufferUsageFlags MEAN_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    using MeanLayout = layout::PrimitiveLayout<weight_type, storageQualifier>; 
    using MeanView = layout::BufferView<MeanLayout>;

    static constexpr vk::DeviceSize minMeanBufferSize(size_t sizeOfWeight) {
        return sizeOfWeight;
    }

    /**
     * Buffer which contains the resulting splits!
     */
    merian::BufferHandle splits;
    static constexpr vk::BufferUsageFlags SPLITS_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    using SplitStructLayout = layout::StructLayout<storageQualifier,
          layout::Attribute<wrs::glsl::uint, "i">,
          layout::Attribute<wrs::glsl::uint, "j">,
          layout::Attribute<weight_type, "spill">>;
    using SplitsLayout = layout::ArrayLayout<SplitStructLayout, storageQualifier>;
    using SplitsView = layout::BufferView<SplitsLayout>;

    static constexpr vk::DeviceSize minSplitBufferSize(uint32_t K, size_t sizeOfWeight) {
        vk::DeviceSize sizeOfSplitDescriptor = sizeof(uint32_t) + sizeof(uint32_t) + sizeOfWeight;
        return sizeOfSplitDescriptor * K;
    }

    template <std::floating_point weight_t> using Split = wrs::Split<weight_t, wrs::glsl::uint>;

    static ScalarSplitBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                         std::size_t weightCount,
                                         std::size_t splitCount,
                                         merian::MemoryMappingType memoryMapping);

};

class ScalarSplit {
  public:
    using weight_t = float;

    ScalarSplit(const merian::ContextHandle& context, glsl::uint workgroupSize = 512) : m_workgroupSize(workgroupSize) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // partition prefix sums
                .add_binding_storage_buffer() // mean
                .add_binding_storage_buffer() // splits
                .build_push_descriptor_layout(context);
        std::string shaderPath = "src/wrs/algorithm/split/scalar/float.comp";
        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<std::tuple<uint32_t, uint32_t>>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);

        m_writes.resize(3);
        vk::WriteDescriptorSet& partitionPrefix = m_writes[0];
        partitionPrefix.setDstBinding(0);
        partitionPrefix.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& mean = m_writes[1];
        mean.setDstBinding(1);
        mean.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& splits = m_writes[2];
        splits.setDstBinding(2);
        splits.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    }

    void run(const vk::CommandBuffer cmd, const ScalarSplitBuffers& buffers, uint32_t N, uint32_t K) {
        m_pipeline->bind(cmd);

        vk::DescriptorBufferInfo prefixDesc = buffers.partitionPrefix->get_descriptor_info();
        m_writes[0].setBufferInfo(prefixDesc);
        vk::DescriptorBufferInfo meanDesc = buffers.mean->get_descriptor_info();
        m_writes[1].setBufferInfo(meanDesc);
        vk::DescriptorBufferInfo splitDesc = buffers.splits->get_descriptor_info();
        m_writes[2].setBufferInfo(splitDesc);
        m_pipeline->push_descriptor_set(cmd, m_writes);

        // NOTE: tuples are stored in reverse order by entries (makes it a bit weird when mapping)
        m_pipeline->push_constant<std::tuple<uint32_t, uint32_t>>(cmd, std::make_tuple(N, K));
        
        glsl::uint workgroupCount = (K - 1 + m_workgroupSize - 1) / m_workgroupSize;

        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    std::vector<vk::WriteDescriptorSet> m_writes;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
