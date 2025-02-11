#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <bit>
#include <concepts>
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct SubgroupPackBuffers {
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    using weight_type = glsl::f32;

    merian::BufferHandle partitionIndices; // bind = 0
    using PartitionIndicesLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "heavyCount">,
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

    merian::BufferHandle partition; // binding = 5
    using PartitionLayout = layout::ArrayLayout<float, storageQualifier>;
    using PartitionView = layout::BufferView<PartitionLayout>;

    merian::BufferHandle partitionPrefix; // binding = 6
    using PartitionPrefixLayout = layout::ArrayLayout<float, storageQualifier>;
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout>;

    static SubgroupPackBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                        std::size_t weightCount,
                                        std::size_t splitCount,
                                        merian::MemoryMappingType memoryMapping);
};

struct SubgroupPackConfig {
    const glsl::uint splitSize; // K
    const glsl::uint workgroupSize;
    const glsl::uint subgroupSplit;

    constexpr SubgroupPackConfig() : splitSize(2), workgroupSize(512), subgroupSplit(4) {}
    explicit constexpr SubgroupPackConfig(glsl::uint splitSize, glsl::uint workgroupSize,
        glsl::uint subgroupSplit)
        : splitSize(splitSize), workgroupSize(workgroupSize), subgroupSplit(subgroupSplit) {}
};

class SubgroupPack {
    struct PushConstants {
        glsl::uint N;
        glsl::uint K;
    };

  public:
    using Buffers = SubgroupPackBuffers;

    explicit SubgroupPack(const merian::ContextHandle& context, SubgroupPackConfig config = {})
        : m_splitSize(config.splitSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // partition indices
                .add_binding_storage_buffer() // weights
                .add_binding_storage_buffer() // mean
                .add_binding_storage_buffer() // splits
                .add_binding_storage_buffer() // alias table
                .add_binding_storage_buffer() // partition
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/pack/subgroup/shader.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(config.workgroupSize); // 0
        glsl::uint subgroupSize =
            context->physical_device.physical_device_subgroup_properties.subgroupSize;
        assert(config.workgroupSize % subgroupSize == 0);
        specInfoBuilder.add_entry(subgroupSize); // 1
        /* glsl::uint log2SubgroupSize = std::bit_width(subgroupSize) - 1; // floor(log2( . )) */
        /* specInfoBuilder.add_entry(log2SubgroupSize); */
        
        glsl::uint threadsPerSubproblem = subgroupSize / config.subgroupSplit;
        specInfoBuilder.add_entry(threadsPerSubproblem); // 2
        glsl::uint log2ThreadsPerSubgroup = std::bit_width(threadsPerSubproblem) - 1;
        specInfoBuilder.add_entry(log2ThreadsPerSubgroup); // 3

        specInfoBuilder.add_entry(config.splitSize); // 4

        glsl::uint subgroupCount = (config.workgroupSize + subgroupSize - 1) / subgroupSize;
        m_subproblemsPerWorkgroup = subgroupCount * config.subgroupSplit;

        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, glsl::uint N) {

        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.partitionIndices, buffers.weights,
                                        buffers.mean, buffers.splits, buffers.aliasTable, buffers.partition);

        glsl::uint K = ((N + m_splitSize - 1) / m_splitSize);
        m_pipeline->push_constant<PushConstants>(cmd, PushConstants{
                                                          .N = N,
                                                          .K = K,
                                                      });
        const uint32_t workgroupCount = (K + m_subproblemsPerWorkgroup - 1) / m_subproblemsPerWorkgroup;
        fmt::println("DISPATCH-WORKGROUP-COUNT: {}    K ={}", workgroupCount, K);
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_subproblemsPerWorkgroup;
    glsl::uint m_splitSize;
};

} // namespace wrs
