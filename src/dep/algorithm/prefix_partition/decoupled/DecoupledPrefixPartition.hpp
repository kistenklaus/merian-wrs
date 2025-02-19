#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

template <typename T>
concept decoupled_prefix_partition_compatible = std::same_as<float, T>;

struct DecoupledPrefixPartitionBuffers {
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    using Self = DecoupledPrefixPartitionBuffers;

    merian::BufferHandle elements;

    template <decoupled_prefix_partition_compatible T>
    using ElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using ElementsView = layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <decoupled_prefix_partition_compatible T>
    using PivotLayout = layout::PrimitiveLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using PivotView = layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle decoupledStates;
    template <decoupled_prefix_partition_compatible T>
    using _DecoupledState =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "heavyCount">,
                             layout::Attribute<glsl::uint, "heavyCountInclusivePrefix">,
                             layout::Attribute<T, "heavySum">,
                             layout::Attribute<T, "heavyInclusivePrefix">,
                             layout::Attribute<T, "lightSum">,
                             layout::Attribute<T, "lightInclusivePrefix">,
                             layout::Attribute<glsl::uint, "state">>;
    template <decoupled_prefix_partition_compatible T>
    using _DecoupledStateArray = layout::ArrayLayout<_DecoupledState<T>, storageQualifier>;

    template <decoupled_prefix_partition_compatible T>
    using DecoupledStatesLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "atomicBatchCounter">,
                             layout::Attribute<_DecoupledStateArray<T>, "partitions">>;
    template <decoupled_prefix_partition_compatible T>
    using DecoupledStatesView = layout::BufferView<DecoupledStatesLayout<T>>;

    merian::BufferHandle heavyCount;
    template <decoupled_prefix_partition_compatible T>
    using HeavyCountLayout = layout::PrimitiveLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using HeavyCountView = layout::BufferView<HeavyCountLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <decoupled_prefix_partition_compatible T>
    using PartitionElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using PartitionElementsView = layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle partitionPrefix;
    template <decoupled_prefix_partition_compatible T>
    using PartitionPrefixLayout = layout::ArrayLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout<T>>;


    template <decoupled_prefix_partition_compatible T>
    static DecoupledPrefixPartitionBuffers allocate(const merian::ResourceAllocatorHandle& alloc,
                                                    merian::MemoryMappingType memoryMapping,
                                                    std::size_t N,
                                                    std::size_t blockCount);
};

template <>
DecoupledPrefixPartitionBuffers
DecoupledPrefixPartitionBuffers::allocate<float>(const merian::ResourceAllocatorHandle& alloc,
                                                 merian::MemoryMappingType memoryMapping,
                                                 std::size_t N,
                                                 std::size_t blockCount);


class DecoupledPrefixPartitionConfig {
  public:
    glsl::uint workgroupSize;
    glsl::uint rows;
    glsl::uint parallelLookbackDepth;
    BlockScanVariant blockScanVariant;

    constexpr DecoupledPrefixPartitionConfig()
        : workgroupSize(512), rows(8), parallelLookbackDepth(32),
          blockScanVariant(BlockScanVariant::RANKED_STRIDED) {}
    explicit constexpr DecoupledPrefixPartitionConfig(glsl::uint workgroupSize,
                                                      glsl::uint rows,
                                                      glsl::uint parallelLookbackDepth,
                                                      BlockScanVariant variant)
        : workgroupSize(workgroupSize), rows(rows), parallelLookbackDepth(parallelLookbackDepth),
          blockScanVariant(variant) {}

    constexpr glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

template <decoupled_prefix_partition_compatible T>
class DecoupledPrefixPartition {
  public:
    using weight_t = float;
    using Buffers = DecoupledPrefixPartitionBuffers;

    explicit DecoupledPrefixPartition(const merian::ContextHandle& context,
                                      const merian::ShaderCompilerHandle& shaderCompiler,
                                      DecoupledPrefixPartitionConfig config = {})
        : m_blockSize(config.blockSize()) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // pivot
                .add_binding_storage_buffer() // decoupled states
                .add_binding_storage_buffer() // heavy count
                .add_binding_storage_buffer() // partition indices
                .add_binding_storage_buffer() // partition prefix
                .add_binding_storage_buffer() // partition elements
                .build_push_descriptor_layout(context);

        std::string shaderPath = "src/wrs/algorithm/prefix_partition/decoupled/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        assert(context->physical_device.physical_device_subgroup_properties.subgroupSize >=
               config.parallelLookbackDepth);
        specInfoBuilder.add_entry(config.workgroupSize);
        specInfoBuilder.add_entry(context->physical_device.physical_device_subgroup_properties.subgroupSize);
        specInfoBuilder.add_entry(config.rows);
        specInfoBuilder.add_entry(config.parallelLookbackDepth);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }
    void run(const merian::CommandBufferHandle& cmd,
             const DecoupledPrefixPartitionBuffers& buffers,
             uint32_t N) {
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot,
                                 buffers.decoupledStates, buffers.heavyCount, buffers.partitionIndices,
                                 buffers.partitionPrefix, buffers.partitionElements);

        cmd->push_constant(m_pipeline, N);
        uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint blockSize() const {
        return m_blockSize;
    }

  private:
    const uint32_t m_blockSize;
    merian::PipelineHandle m_pipeline;
};

} // namespace wrs
