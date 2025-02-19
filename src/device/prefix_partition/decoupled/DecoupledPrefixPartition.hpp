#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/prefix_partition/PrefixPartitionAllocFlags.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/Attribute.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/layout/PrimitiveLayout.hpp"
#include "src/host/layout/StructLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

template <typename T>
concept decoupled_prefix_partition_compatible = std::same_as<float, T>;

struct DecoupledPrefixPartitionBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using Self = DecoupledPrefixPartitionBuffers;

    merian::BufferHandle elements;

    template <decoupled_prefix_partition_compatible T>
    using ElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using ElementsView = host::layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <decoupled_prefix_partition_compatible T>
    using PivotLayout = host::layout::PrimitiveLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using PivotView = host::layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle decoupledStates;
    template <decoupled_prefix_partition_compatible T>
    using _DecoupledState = host::layout::StructLayout<
        storageQualifier,
        host::layout::Attribute<host::glsl::uint, "heavyCount">,
        host::layout::Attribute<host::glsl::uint, "heavyCountInclusivePrefix">,
        host::layout::Attribute<T, "heavySum">,
        host::layout::Attribute<T, "heavyInclusivePrefix">,
        host::layout::Attribute<T, "lightSum">,
        host::layout::Attribute<T, "lightInclusivePrefix">,
        host::layout::Attribute<host::glsl::uint, "state">>;
    template <decoupled_prefix_partition_compatible T>
    using _DecoupledStateArray = host::layout::ArrayLayout<_DecoupledState<T>, storageQualifier>;

    template <decoupled_prefix_partition_compatible T>
    using DecoupledStatesLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<host::glsl::uint, "atomicBatchCounter">,
                                   host::layout::Attribute<_DecoupledStateArray<T>, "partitions">>;
    template <decoupled_prefix_partition_compatible T>
    using DecoupledStatesView = host::layout::BufferView<DecoupledStatesLayout<T>>;

    merian::BufferHandle heavyCount;
    template <decoupled_prefix_partition_compatible T>
    using HeavyCountLayout = host::layout::PrimitiveLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <decoupled_prefix_partition_compatible T>
    using PartitionElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using PartitionElementsView = host::layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle partitionPrefix;
    template <decoupled_prefix_partition_compatible T>
    using PartitionPrefixLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <decoupled_prefix_partition_compatible T>
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout<T>>;

    template <decoupled_prefix_partition_compatible T>
    static DecoupledPrefixPartitionBuffers
    allocate(const merian::ResourceAllocatorHandle& alloc,
             merian::MemoryMappingType memoryMapping,
             std::size_t N,
             std::size_t blockCount,
             PrefixPartitionAllocFlags allocFlags = PrefixPartitionAllocFlags::ALLOC_ALL);
};

template <>
DecoupledPrefixPartitionBuffers
DecoupledPrefixPartitionBuffers::allocate<float>(const merian::ResourceAllocatorHandle& alloc,
                                                 merian::MemoryMappingType memoryMapping,
                                                 std::size_t N,
                                                 std::size_t blockCount,
                                                 PrefixPartitionAllocFlags allocFlags);

class DecoupledPrefixPartitionConfig {
  public:
    host::glsl::uint workgroupSize;
    host::glsl::uint rows;
    host::glsl::uint parallelLookbackDepth;
    BlockScanVariant blockScanVariant;

    constexpr DecoupledPrefixPartitionConfig()
        : workgroupSize(512), rows(8), parallelLookbackDepth(32),
          blockScanVariant(BlockScanVariant::RANKED_STRIDED) {}
    explicit constexpr DecoupledPrefixPartitionConfig(host::glsl::uint workgroupSize,
                                                      host::glsl::uint rows,
                                                      BlockScanVariant variant,
                                                      host::glsl::uint parallelLookbackDepth = 32)
        : workgroupSize(workgroupSize), rows(rows), parallelLookbackDepth(parallelLookbackDepth),
          blockScanVariant(variant) {}

    constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

template <decoupled_prefix_partition_compatible T> class DecoupledPrefixPartition {
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

        std::string shaderPath = "src/device/prefix_partition/decoupled/shader.comp";

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
        specInfoBuilder.add_entry(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
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
                                 buffers.decoupledStates, buffers.heavyCount,
                                 buffers.partitionIndices, buffers.partitionPrefix,
                                 buffers.partitionElements);

        cmd->push_constant(m_pipeline, N);
        uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline host::glsl::uint blockSize() const {
        return m_blockSize;
    }

  private:
    const uint32_t m_blockSize;
    merian::PipelineHandle m_pipeline;
};

} // namespace device
