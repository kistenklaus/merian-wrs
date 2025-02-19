#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

namespace device {

struct PartitionCombineBuffers {
    using Self = PartitionCombineBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = host::layout::BufferView<ElementsLayout>;

    merian::BufferHandle pivot;
    using PivotLayout = host::layout::PrimitiveLayout<float, storageQualifier>;
    using PivotView = host::layout::BufferView<PivotLayout>;

    merian::BufferHandle indices;
    using IndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using IndicesView = host::layout::BufferView<IndicesLayout>;

    merian::BufferHandle blockIndices;
    using BlockIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using BlockIndicesView = host::layout::BufferView<BlockIndicesLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partition;
    using PartitionLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using PartitionView = host::layout::BufferView<PartitionLayout>;
};

struct PartitionCombineConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;
    const host::glsl::uint sequentialCombineLength;
    const host::glsl::uint blocksPerWorkgroup;

    constexpr PartitionCombineConfig(host::glsl::uint workgroupSize,
                                     host::glsl::uint rows,
                                     host::glsl::uint sequentialCombineLength,
                                     host::glsl::uint blocksPerWorkgroup)
        : workgroupSize(workgroupSize), rows(rows),
          sequentialCombineLength(sequentialCombineLength), blocksPerWorkgroup(blocksPerWorkgroup) {
    }

    constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows * sequentialCombineLength;
    }

    inline host::glsl::uint tileSize() const {
        return blockSize() * blocksPerWorkgroup;
    }
};

class PartitionCombine {
    struct PushConstants {
      host::glsl::uint N;
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

        const std::string shaderPath = "src/device/partition/block_wise/combine/shader.comp";

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

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot, buffers.indices,
                                 buffers.blockIndices, buffers.partitionIndices, buffers.partition);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_tileSize - 1) / m_tileSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline host::glsl::uint tileSize() const {
        return m_tileSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_tileSize;
};

} // namespace wrs
