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
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <fmt/base.h>
#include <stdexcept>

namespace wrs {

struct BlockWisePrefixPartitionBlockScanBuffers {
    using Self = BlockWisePrefixPartitionBlockScanBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    merian::BufferHandle elements;
    template <block_wise_prefix_partition_compatible T>
    using ElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using ElementsView = layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <block_wise_prefix_partition_compatible T>
    using PivotLayout = layout::PrimitiveLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T> //
    using PivotView = layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle blockHeavyCount;
    using BlockHeavyCountLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using BlockHeavyCountView = layout::BufferView<BlockHeavyCountLayout>;

    merian::BufferHandle blockHeavyReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsView = layout::BufferView<BlockHeavyReductionsLayout<T>>;

    merian::BufferHandle blockLightReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsView = layout::BufferView<BlockLightReductionsLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <block_wise_prefix_partition_compatible T>
    using PartitionElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using PartitionElementsView = layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle partitionPrefix;
    template <block_wise_prefix_partition_compatible T>
    using PartitionPrefixLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout<T>>;

    template <block_wise_prefix_partition_compatible T>
    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N,
                         glsl::uint blockCount) {
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

            buffers.partitionIndices = alloc->createBuffer(
                PartitionIndicesLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                memoryMapping);
            buffers.partitionElements = alloc->createBuffer(
                PartitionElementsLayout<T>::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                memoryMapping);
            buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                                          vk::BufferUsageFlagBits::eStorageBuffer |
                                                              vk::BufferUsageFlagBits::eTransferSrc,
                                                          memoryMapping);
        } else {
            buffers.elements = alloc->createBuffer(
                ElementsLayout<T>::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.pivot = alloc->createBuffer(
                PivotLayout<T>::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.blockHeavyCount = nullptr;
            buffers.blockHeavyReductions = nullptr;
            buffers.blockLightReductions = nullptr;

            buffers.partitionIndices =
                alloc->createBuffer(PartitionIndicesLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.partitionElements =
                alloc->createBuffer(PartitionElementsLayout<T>::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.partitionPrefix =
                alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
    }
};

struct BlockWisePrefixPartitionBlockScanConfig {
    const glsl::uint workgroupSize;
    const glsl::uint rows;
    const glsl::uint sequentialScanLength;
    const BlockScanVariant variant;

    constexpr BlockWisePrefixPartitionBlockScanConfig(glsl::uint workgroupSize,
                                                      glsl::uint rows,
                                                      glsl::uint sequentialScanLength,
                                                      BlockScanVariant variant)
        : workgroupSize(workgroupSize), rows(rows), sequentialScanLength(sequentialScanLength),
          variant(variant) {}

    inline constexpr glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

template <block_wise_prefix_partition_compatible T> class BlockWisePrefixPartitionBlockScan {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = BlockWisePrefixPartitionBlockScanBuffers;

    explicit BlockWisePrefixPartitionBlockScan(const merian::ContextHandle& context,
                                               const merian::ShaderCompilerHandle& shaderCompiler,
                                               BlockWisePrefixPartitionBlockScanConfig config)
        : m_blockSize(config.blockSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // pivot (condition)
                .add_binding_storage_buffer() // block heavy count
                .add_binding_storage_buffer() // block heavy reductions
                .add_binding_storage_buffer() // block light reductions
                .add_binding_storage_buffer() // partition Indices
                .add_binding_storage_buffer() // partition Prefix
                .add_binding_storage_buffer() // partition Elements
                .build_push_descriptor_layout(context);

        const std::string shaderPath =
            "src/wrs/algorithm/prefix_partition/block_wise/block_scan/shader.comp";

        std::map<std::string, std::string> defines;

        if ((config.variant & BlockScanVariant::RANKED) == BlockScanVariant::RANKED) {
            defines["BLOCK_SCAN_USE_RANKED"];
        } else if ((config.variant & BlockScanVariant::RAKING) == BlockScanVariant::RAKING) {
            defines["BLOCK_SCAN_USE_RAKING"];
        }
        if ((config.variant & BlockScanVariant::SUBGROUP_SCAN_SHFL) ==
            BlockScanVariant::SUBGROUP_SCAN_SHFL) {
            defines["SUBGROUP_SCAN_USE_SHFL"];
        }
        if ((config.variant & BlockScanVariant::EXCLUSIVE) == BlockScanVariant::EXCLUSIVE) {
            defines["EXCLUSIVE"];
        } 
        if ((config.variant & BlockScanVariant::INCLUSIVE) == BlockScanVariant::INCLUSIVE) {
            defines["INCLUSIVE"];
        }

        if ((config.variant & BlockScanVariant::STRIDED) == BlockScanVariant::STRIDED) {
            if ((config.variant & BlockScanVariant::RAKING) == BlockScanVariant::RAKING) {
                throw std::runtime_error("Unsupported variant");
            }
            defines["STRIDED"];
        }

        if constexpr (std::same_as<T, glsl::f32>) {
            defines["USE_FLOAT"]; // NOTE only support float partitions
        } else if constexpr (std::same_as<T, glsl::uint>) {
            defines["USE_UINT"];
        } else {
            throw std::runtime_error("Uncompatible base type");
        }

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {"src/wrs/algorithm/include"},
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

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, glsl::uint N) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot,
                                 buffers.blockHeavyCount, buffers.blockHeavyReductions,
                                 buffers.blockLightReductions, buffers.partitionIndices,
                                 buffers.partitionPrefix, buffers.partitionElements);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;
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
