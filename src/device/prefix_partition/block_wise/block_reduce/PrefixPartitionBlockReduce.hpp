#pragma once

#include "../compatible.hpp"

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <concepts>
#include <stdexcept>

namespace device {

struct BlockWisePrefixPartitionBlockReduceBuffers {
    using Self = BlockWisePrefixPartitionBlockReduceBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    merian::BufferHandle elements;
    template <block_wise_prefix_partition_compatible T>
    using ElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using ElementsView = host::layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <block_wise_prefix_partition_compatible T>
    using PivotLayout = host::layout::PrimitiveLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T> //
    using PivotView = host::layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle blockHeavyCount;
    using BlockHeavyCountLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using BlockHeavyCountView = host::layout::BufferView<BlockHeavyCountLayout>;

    merian::BufferHandle blockHeavyReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsView = host::layout::BufferView<BlockHeavyReductionsLayout<T>>;

    merian::BufferHandle blockLightReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsView = host::layout::BufferView<BlockLightReductionsLayout<T>>;

    template <block_wise_prefix_partition_compatible T>
    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N,
                         host::glsl::uint blockCount) {
        Self buffers;

        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.elements = alloc->createBuffer(ElementsLayout<T>::size(N),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferDst,
                                                   memoryMapping);
            buffers.pivot = alloc->createBuffer(PivotLayout<T>::size(),
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                memoryMapping);
            buffers.blockHeavyCount =
                alloc->createBuffer(BlockHeavyCountLayout::size(blockCount),
                                    vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            buffers.blockHeavyReductions =
                alloc->createBuffer(BlockHeavyReductionsLayout<T>::size(blockCount),
                                    vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            buffers.blockLightReductions =
                alloc->createBuffer(BlockLightReductionsLayout<T>::size(blockCount),
                                    vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
        } else {
            buffers.elements = alloc->createBuffer(
                ElementsLayout<T>::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.pivot = alloc->createBuffer(
                PivotLayout<T>::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.blockHeavyCount = nullptr;
            buffers.blockHeavyReductions = nullptr;
            buffers.blockLightReductions = nullptr;
        }
    }
};

struct BlockWisePrefixPartitionBlockReduceConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;

    constexpr BlockWisePrefixPartitionBlockReduceConfig(host::glsl::uint workgroupSize,
                                                        host::glsl::uint rows)
        : workgroupSize(workgroupSize), rows(rows) {}

    inline constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

template <block_wise_prefix_partition_compatible T> class BlockWisePrefixPartitionBlockReduce {
    struct PushConstants {
        host::glsl::uint N;
    };

  public:
    using Buffers = BlockWisePrefixPartitionBlockReduceBuffers;

    explicit BlockWisePrefixPartitionBlockReduce(const merian::ContextHandle& context,
                                                 const merian::ShaderCompilerHandle& shaderCompiler,
                                                 BlockWisePrefixPartitionBlockReduceConfig config)
        : m_blockSize(config.blockSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // pivot (condition)
                .add_binding_storage_buffer() // block heavy count
                .add_binding_storage_buffer() // block heavy reductions
                .add_binding_storage_buffer() // block light reductions
                .build_push_descriptor_layout(context);

        const std::string shaderPath =
            "src/device/prefix_partition/block_wise/block_reduce/shader.comp";

        std::map<std::string, std::string> defines;

        if constexpr (std::same_as<T, host::glsl::f32>) {
            defines["USE_FLOAT"]; // NOTE only support float partitions
        } else if constexpr (std::same_as<T, host::glsl::uint>) {
            defines["USE_UINT"];
        } else {
            throw std::runtime_error("Uncompatible base type");
        }

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {"src/device/common/"},
            defines);

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
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot,
                                 buffers.blockHeavyCount, buffers.blockHeavyReductions,
                                 buffers.blockLightReductions);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline host::glsl::uint blockSize() const {
        return m_blockSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_blockSize;
};

} // namespace device
