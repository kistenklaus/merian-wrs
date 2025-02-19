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
#include <concepts>
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct PartitionCombineBuffers {
    using Self = PartitionCombineBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    merian::BufferHandle pivot;
    using PivotLayout = layout::PrimitiveLayout<float, storageQualifier>;
    using PivotView = layout::BufferView<PivotLayout>;

    merian::BufferHandle indices;
    using IndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using IndicesView = layout::BufferView<IndicesLayout>;

    merian::BufferHandle blockIndices;
    using BlockIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using BlockIndicesView = layout::BufferView<BlockIndicesLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partition;
    using PartitionLayout = layout::ArrayLayout<float, storageQualifier>;
    using PartitionView = layout::BufferView<PartitionLayout>;
};

struct PartitionCombineConfig {
    const glsl::uint workgroupSize;
    const glsl::uint rows;
    const glsl::uint sequentialCombineLength;
    const glsl::uint blocksPerWorkgroup;

    constexpr PartitionCombineConfig(glsl::uint workgroupSize,
                                     glsl::uint rows,
                                     glsl::uint sequentialCombineLength,
                                     glsl::uint blocksPerWorkgroup)
        : workgroupSize(workgroupSize), rows(rows),
          sequentialCombineLength(sequentialCombineLength), blocksPerWorkgroup(blocksPerWorkgroup) {
    }

    constexpr glsl::uint blockSize() const {
        return workgroupSize * rows * sequentialCombineLength;
    }

    inline glsl::uint tileSize() const {
        return blockSize() * blocksPerWorkgroup;
    }
};

class PartitionCombine {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = PartitionCombineBuffers;

    explicit PartitionCombine(const merian::ContextHandle& context,
                              const merian::ShaderCompilerHandle& shaderCompiler,
                              PartitionCombineConfig config)
        : m_tileSize(config.tileSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // pivot (condition)
                .add_binding_storage_buffer() // indices
                .add_binding_storage_buffer() // blockIndices
                .add_binding_storage_buffer() // partitionIndices
                .add_binding_storage_buffer() // partition
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/partition/block_wise/combine/shader.comp";

        std::map<std::string, std::string> defines;
        defines["USE_FLOAT"]; // currently only supports partition of floating values

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {}, defines);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(config.workgroupSize);
        specInfoBuilder.add_entry(config.rows);
        specInfoBuilder.add_entry(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        specInfoBuilder.add_entry(config.sequentialCombineLength);
        specInfoBuilder.add_entry(config.blocksPerWorkgroup);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, glsl::uint N) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot, buffers.indices,
                                 buffers.blockIndices, buffers.partitionIndices, buffers.partition);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_tileSize - 1) / m_tileSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint tileSize() const {
        return m_tileSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_tileSize;
};

} // namespace wrs
