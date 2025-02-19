#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
namespace wrs {

struct ScalarPackBuffers {
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    using weight_type = glsl::f32;

    merian::BufferHandle partitionIndices; // bind = 0
    using PartitionIndicesLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "heavyCount">,
                             layout::Attribute<glsl::uint, "__padding1">,
                             layout::Attribute<glsl::uint, "__padding2">,
                             layout::Attribute<glsl::uint, "__padding3">,
                             layout::Attribute<glsl::uint*, "heavyLightIndices">>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle weights; // binding = 1
    using WeightsLayout = layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean; // binding = 2
    using MeanLayout = layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = layout::BufferView<MeanLayout>;

    merian::BufferHandle splits; // binding = 3
    using SplitStructLayout = layout::StructLayout<storageQualifier,
                                                   layout::Attribute<glsl::uint, "i">,
                                                   layout::Attribute<glsl::uint, "j">,
                                                   layout::Attribute<weight_type, "spill">>;
    using SplitsLayout = layout::ArrayLayout<SplitStructLayout, storageQualifier>;
    using SplitsView = layout::BufferView<SplitsLayout>;

    merian::BufferHandle aliasTable; // binding = 4
    using AliasTableEntryLayout = layout::StructLayout<storageQualifier,
                                                       layout::Attribute<weight_type, "p">,
                                                       layout::Attribute<glsl::uint, "a">>;
    using AliasTableLayout = layout::ArrayLayout<AliasTableEntryLayout, storageQualifier>;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    static ScalarPackBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                      std::size_t weightCount,
                                      std::size_t splitCount,
                                      merian::MemoryMappingType memoryMapping);
};

class ScalarPackConfig {
  public:
    glsl::uint workgroupSize;

    constexpr ScalarPackConfig() : workgroupSize(512) {}
    explicit constexpr ScalarPackConfig(glsl::uint workgroupSize) : workgroupSize(workgroupSize) {}
};

class ScalarPack {

    struct PushConstant {
        glsl::uint N;
        glsl::uint K;
    };

  public:
    using Buffers = ScalarPackBuffers;
    using weight_t = float;

    explicit ScalarPack(const merian::ContextHandle& context,
                        const merian::ShaderCompilerHandle& shaderCompiler,
                        ScalarPackConfig config = {})
        : m_workgroupSize(config.workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // partition indices
                .add_binding_storage_buffer() // weights
                .add_binding_storage_buffer() // mean
                .add_binding_storage_buffer() // splits
                .add_binding_storage_buffer() // alias table
                .build_push_descriptor_layout(context);

        std::string shaderPath = "src/wrs/algorithm/pack/scalar/preload_float.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstant>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry<glsl::uint>(config.workgroupSize);
        specInfoBuilder.add_entry<glsl::uint>(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const glsl::uint N,
             const glsl::uint K,
             const ScalarPackBuffers& buffers) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.partitionIndices, buffers.weights,
                                 buffers.mean, buffers.splits, buffers.aliasTable);
        cmd->push_constant<PushConstant>(m_pipeline, PushConstant{.N = N, .K = K});
        const uint32_t workgroupCount = (K + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
