#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct DecoupledPartitionBuffers {
    using Self = DecoupledPartitionBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    // ======  INPUT =========
    merian::BufferHandle elements;
    using ElementsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    merian::BufferHandle pivot;
    using PivotLayout = layout::PrimitiveLayout<float, storageQualifier>;
    using PivotView = layout::BufferView<PivotLayout>;

    // ======= INTERMEDIATE =============

    merian::BufferHandle decoupledStates;
    using _DecoupledStateLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<float, layout::StaticString("aggregate")>,
                             layout::Attribute<float, layout::StaticString("prefix")>,
                             layout::Attribute<glsl::uint, layout::StaticString("state")>>;
    using _DecoupledStatesArrayLayout =
        layout::ArrayLayout<_DecoupledStateLayout, storageQualifier>;
    using DecoupledStatesLayout = layout::StructLayout<
        storageQualifier,
        layout::Attribute<glsl::uint, layout::StaticString("counter")>,
        layout::Attribute<_DecoupledStatesArrayLayout, layout::StaticString("batches")>>;
    using DecoupledStatesView = layout::BufferView<DecoupledStatesLayout>;

    // ======== OUTPUT =============
    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partition;
    using PartitionLayout = layout::ArrayLayout<float, storageQualifier>;
    using PartitionView = layout::BufferView<PartitionLayout>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = layout::PrimitiveLayout<glsl::uint, storageQualifier>;
    using HeavyCountView = layout::BufferView<HeavyCountLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N,
                         glsl::uint blockCount);
};

struct DecoupledPartitionConfig {
    const glsl::uint workgroupSize;
    const glsl::uint rows;
    const BlockScanVariant blockScanVariant;
    const glsl::uint parallelLookbackDepth;

    constexpr DecoupledPartitionConfig(glsl::uint workgroupSize,
                                       glsl::uint rows,
                                       BlockScanVariant blockScanVariant,
                                       glsl::uint parallelLookbackDepth)
        : workgroupSize(workgroupSize), rows(rows), blockScanVariant(blockScanVariant),
          parallelLookbackDepth(parallelLookbackDepth) {}

    glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

class DecoupledPartition {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = DecoupledPartitionBuffers;

    explicit DecoupledPartition(const merian::ContextHandle& context,
        const merian::ShaderCompilerHandle& shaderCompiler,
                                DecoupledPartitionConfig config)
        : m_blockSize(config.blockSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // pivot
                .add_binding_storage_buffer() // decoupled states
                .add_binding_storage_buffer() // partition indices
                .add_binding_storage_buffer() // partition elements
                .add_binding_storage_buffer() // heavy count
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/partition/decoupled/shader.comp";

        std::map<std::string, std::string> defines;
        if ((config.blockScanVariant & BlockScanVariant::RANKED) == BlockScanVariant::RANKED) {
            defines["BLOCK_SCAN_USE_RANKED"];
        } else if ((config.blockScanVariant & BlockScanVariant::RAKING) ==
                   BlockScanVariant::RAKING) {
            defines["BLOCK_SCAN_USE_RAKING"];
        }
        if ((config.blockScanVariant & BlockScanVariant::SUBGROUP_SCAN_SHFL) ==
            BlockScanVariant::SUBGROUP_SCAN_SHFL) {
            defines["SUBGROUP_SCAN_USE_SHFL"];
        }
        if ((config.blockScanVariant & BlockScanVariant::EXCLUSIVE) ==
            BlockScanVariant::EXCLUSIVE) {
            defines["EXCLUSIVE"];
        }
        if ((config.blockScanVariant & BlockScanVariant::INCLUSIVE) ==
            BlockScanVariant::INCLUSIVE) {
            const auto it = defines.find("EXCLUSIVE");
            if (it != defines.end()) {
                defines.erase(it);
            }
        }
        if ((config.blockScanVariant & BlockScanVariant::STRIDED) == BlockScanVariant::STRIDED) {
            if ((config.blockScanVariant & BlockScanVariant::RAKING) == BlockScanVariant::RAKING) {
                throw std::runtime_error("Unsupported BlockScanVariant");
            }
            defines["STRIDED"];
        }
        defines["USE_FLOAT"];

        const merian::ShaderModuleHandle shader =
          shaderCompiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute,
                {"src/wrs/algorithm/include/"}, defines);

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
        specInfoBuilder.add_entry(config.parallelLookbackDepth);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, glsl::uint N) {

        const uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;

        // Didn't find any smart way around this, but zeroing buffers is pretty fast and
        // most likely hardware accelerated (faster than memory bandwidth)
        Buffers::DecoupledStatesView decoupledStatesView{buffers.decoupledStates, workgroupCount};
        decoupledStatesView.zero(cmd);
        decoupledStatesView.expectComputeRead(cmd);

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot,
                                        buffers.decoupledStates, buffers.partitionIndices,
                                        buffers.partition, buffers.heavyCount);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});

        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint blockSize() const {
      return m_blockSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_blockSize;
};

} // namespace wrs
