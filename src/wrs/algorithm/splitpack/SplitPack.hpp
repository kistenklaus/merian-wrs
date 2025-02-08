#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct SplitPackBuffers {
    using Self = SplitPackBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "heavyCount">,
                             layout::Attribute<glsl::uint*, "heavyLight">>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<wrs::glsl::uint, layout::StaticString("heavyCount")>,
                             layout::Attribute<float*, layout::StaticString("heavyLight")>>;
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle mean;
    using MeanLayout = layout::PrimitiveLayout<float, storageQualifier>;
    using MeanView = layout::BufferView<MeanLayout>;

    merian::BufferHandle aliasTable;

    using _AliasTableEntryLayout = layout::StructLayout<storageQualifier,
                                                        layout::Attribute<float, "p">,
                                                        layout::Attribute<glsl::uint, "a">>;
    using AliasTableLayout = layout::ArrayLayout<_AliasTableEntryLayout, storageQualifier>;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    merian::BufferHandle splits;
    using SplitStructLayout = layout::StructLayout<storageQualifier,
                                                   layout::Attribute<wrs::glsl::uint, "i">,
                                                   layout::Attribute<wrs::glsl::uint, "j">,
                                                   layout::Attribute<float, "spill">>;
    using SplitsLayout = layout::ArrayLayout<SplitStructLayout, storageQualifier>;
    using SplitsView = layout::BufferView<SplitsLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N);
};

class SplitPack {
    struct PushConstants {
        glsl::uint K;
        glsl::uint N;
    };

  public:
    using Buffers = SplitPackBuffers;

    explicit SplitPack(const merian::ContextHandle& context, glsl::uint workgroupSize,
        glsl::uint splitSize)
        : m_workgroupSize(workgroupSize), m_splitSize(splitSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // weights
                .add_binding_storage_buffer() // partition indices
                .add_binding_storage_buffer() // partition prefix
                .add_binding_storage_buffer() // mean
                .add_binding_storage_buffer() // alias table
                .add_binding_storage_buffer() // debug
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/splitpack/shader_using_weights.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        specInfoBuilder.add_entry(context->physical_device.physical_device_subgroup_properties.subgroupSize);
        specInfoBuilder.add_entry(splitSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, glsl::uint N) {

        glsl::uint K = N / m_splitSize;

        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.weights, buffers.partitionIndices,
                                        buffers.partitionPrefix, buffers.mean, buffers.aliasTable,
                                        buffers.splits);
        m_pipeline->push_constant<PushConstants>(cmd, PushConstants{
                                                          .K = K,
                                                          .N = N,
                                                      });
        const glsl::uint splitsPerDispatch = m_workgroupSize - 1;
        const glsl::uint workgroupCount = (K + splitsPerDispatch - 1) / splitsPerDispatch;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
    glsl::uint m_splitSize;
};

} // namespace wrs
