#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StaticString.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <memory>
#include <vulkan/vulkan_handles.hpp>
#include <cstddef>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct DecoupledPrefixSumBuffers {
    using Self = DecoupledPrefixSumBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    merian::BufferHandle prefixSum;
    using PrefixSumLayout = layout::ArrayLayout<float, storageQualifier>;
    using PrefixSumView = layout::BufferView<PrefixSumLayout>;

    merian::BufferHandle decoupledStates;
    using _DecoupledStateLayout = layout::StructLayout<storageQualifier, 
          layout::Attribute<float, layout::StaticString("aggregate")>,
          layout::Attribute<float, layout::StaticString("prefix")>,
          layout::Attribute<glsl::uint, layout::StaticString("state")>>;
    using _DecoupledStatesArrayLayout = layout::ArrayLayout<_DecoupledStateLayout, storageQualifier>;
    using DecoupledStatesLayout = layout::StructLayout<storageQualifier,
          layout::Attribute<glsl::uint, layout::StaticString("counter")>,
          layout::Attribute<_DecoupledStatesArrayLayout, layout::StaticString("batches")>>;
    using DecoupledStatesView = layout::BufferView<DecoupledStatesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                                    merian::MemoryMappingType memoryMapping, std::size_t N,
                                    std::size_t partitionSize);
};

class DecoupledPrefixSum {
    struct PushConstants {
        glsl::uint N;
    };
  public:
    using Buffers = DecoupledPrefixSumBuffers;

    explicit DecoupledPrefixSum(const merian::ContextHandle& context, glsl::uint workgroupSize,
        glsl::uint rows, glsl::uint parallelLookbackDepth) : m_partitionSize(workgroupSize * rows){

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/prefix_sum/decoupled/shader.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(workgroupSize);
        specInfoBuilder.add_entry(rows);
        specInfoBuilder.add_entry(context->physical_device.physical_device_subgroup_properties.subgroupSize);
        assert(context->physical_device.physical_device_subgroup_properties.subgroupSize >= parallelLookbackDepth);
        specInfoBuilder.add_entry(parallelLookbackDepth);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, glsl::uint N) {

        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.elements, buffers.prefixSum, buffers.decoupledStates);
        m_pipeline->push_constant<PushConstants>(cmd, PushConstants{
                                                          .N = N
        });
        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_partitionSize;
};

}
