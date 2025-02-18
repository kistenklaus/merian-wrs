#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StaticString.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan_handles.hpp>

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

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t partitionSize);
};

class DecoupledPrefixSumConfig {
  public:
    const glsl::uint workgroupSize;
    const glsl::uint rows;
    const glsl::uint parallelLookbackDepth;

    const BlockScanVariant blockScanVariant;

    constexpr DecoupledPrefixSumConfig()
        : workgroupSize(512), rows(8), parallelLookbackDepth(32),
          blockScanVariant(BlockScanVariant::RAKING) {}
    constexpr explicit DecoupledPrefixSumConfig(
        glsl::uint workgroupSize,
        glsl::uint rows,
        glsl::uint parallelLookbackDepth,
        BlockScanVariant blockScanVariant = BlockScanVariant::RANKED_STRIDED)
        : workgroupSize(workgroupSize), rows(rows), parallelLookbackDepth(parallelLookbackDepth),
          blockScanVariant(blockScanVariant) {}

    inline constexpr glsl::uint partitionSize() const {
        return workgroupSize * rows;
    }
};

class DecoupledPrefixSum {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = DecoupledPrefixSumBuffers;

    explicit DecoupledPrefixSum(const merian::ContextHandle& context,
        const merian::ShaderCompilerHandle& shaderCompiler, 
                                DecoupledPrefixSumConfig config = {})
        : m_partitionSize(config.partitionSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/prefix_sum/decoupled/shader.comp";

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
        assert(context->physical_device.physical_device_subgroup_properties.subgroupSize >=
               config.parallelLookbackDepth);
        specInfoBuilder.add_entry(config.parallelLookbackDepth);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle cmd, const Buffers& buffers, glsl::uint N) {
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.prefixSum,
                                        buffers.decoupledStates);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint getPartitionSize() const {
        return m_partitionSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_partitionSize;
};

} // namespace wrs
