#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/partition/PartitionAllocFlags.hpp"
#include "src/device/prefix_sum/PrefixSumAllocFlags.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/Attribute.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/layout/StaticString.hpp"
#include "src/host/layout/StructLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct DecoupledPrefixSumBuffers {
    using Self = DecoupledPrefixSumBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = host::layout::BufferView<ElementsLayout>;

    merian::BufferHandle prefixSum;
    using PrefixSumLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using PrefixSumView = host::layout::BufferView<PrefixSumLayout>;

    merian::BufferHandle decoupledStates;
    using _DecoupledStateLayout = host::layout::StructLayout<
        storageQualifier,
        host::layout::Attribute<float, host::layout::StaticString("aggregate")>,
        host::layout::Attribute<float, host::layout::StaticString("prefix")>,
        host::layout::Attribute<host::glsl::uint, host::layout::StaticString("state")>>;
    using _DecoupledStatesArrayLayout =
        host::layout::ArrayLayout<_DecoupledStateLayout, storageQualifier>;
    using DecoupledStatesLayout = host::layout::StructLayout<
        storageQualifier,
        host::layout::Attribute<host::glsl::uint, host::layout::StaticString("counter")>,
        host::layout::Attribute<_DecoupledStatesArrayLayout,
                                host::layout::StaticString("batches")>>;
    using DecoupledStatesView = host::layout::BufferView<DecoupledStatesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t partitionSize,
                         PrefixSumAllocFlags allocFlags = PrefixSumAllocFlags::ALLOC_ALL);
};

class DecoupledPrefixSumConfig {
  public:
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;
    const host::glsl::uint parallelLookbackDepth;

    const BlockScanVariant blockScanVariant;

    constexpr DecoupledPrefixSumConfig()
        : workgroupSize(512), rows(8), parallelLookbackDepth(32),
          blockScanVariant(BlockScanVariant::RANKED_STRIDED) {}
    constexpr explicit DecoupledPrefixSumConfig(
        host::glsl::uint workgroupSize,
        host::glsl::uint rows,
        BlockScanVariant blockScanVariant = BlockScanVariant::RANKED_STRIDED,
        host::glsl::uint parallelLookbackDepth = 32)
        : workgroupSize(workgroupSize), rows(rows), parallelLookbackDepth(parallelLookbackDepth),
          blockScanVariant(blockScanVariant) {}

    inline constexpr host::glsl::uint partitionSize() const {
        return workgroupSize * rows;
    }
};

class DecoupledPrefixSum {
    struct PushConstants {
        host::glsl::uint N;
    };

    struct ReversePushConstants {
        host::glsl::uint N;          // amount of elements to compute prefix sum for
        host::glsl::uint bufferSize; // size of buffers
    };

  public:
    using Buffers = DecoupledPrefixSumBuffers;

    explicit DecoupledPrefixSum(const merian::ContextHandle& context,
                                const merian::ShaderCompilerHandle& shaderCompiler,
                                DecoupledPrefixSumConfig config = {},
                                bool reverseMemoryOrder = false)
        : m_partitionSize(config.partitionSize()), m_reverseMemoryOrder(reverseMemoryOrder) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        std::string shaderPath;
        if (m_reverseMemoryOrder) {
            shaderPath = "src/device/prefix_sum/decoupled/reverse.comp";
        } else {
            shaderPath = "src/device/prefix_sum/decoupled/shader.comp";
        }

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

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {"src/device/common/"},
            defines);

        auto builder =
            merian::PipelineLayoutBuilder(context).add_descriptor_set_layout(descriptorSet0Layout);
        if (m_reverseMemoryOrder) {
          builder.add_push_constant<ReversePushConstants>();
        }else {
          builder.add_push_constant<PushConstants>();
        }
        const merian::PipelineLayoutHandle pipelineLayout = builder.build_pipeline_layout();
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

    void run(const merian::CommandBufferHandle cmd, const Buffers& buffers, host::glsl::uint N) {

        cmd->fill(buffers.decoupledStates, 0);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.decoupledStates->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                             vk::AccessFlagBits::eShaderRead));

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.prefixSum,
                                 buffers.decoupledStates);
        if (m_reverseMemoryOrder) {
            cmd->push_constant<ReversePushConstants>(
                m_pipeline,
                ReversePushConstants{
                    .N = N,
                    .bufferSize = static_cast<host::glsl::uint>(buffers.elements->get_size() / sizeof(host::glsl::f32)),
                });
        } else {
            cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        }
        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline host::glsl::uint getPartitionSize() const {
        return m_partitionSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    const host::glsl::uint m_partitionSize;
    const bool m_reverseMemoryOrder;
};

} // namespace device
