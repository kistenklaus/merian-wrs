#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct PartitionBlockScanBuffers {
    using Self = PartitionBlockScanBuffers;
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

    merian::BufferHandle blockCount;
    using BlockCountLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using BlockCountView = layout::BufferView<BlockCountLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N,
                         glsl::uint blockCount) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferDst,
                                                   memoryMapping);
            buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                memoryMapping);

            buffers.indices = alloc->createBuffer(IndicesLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);
            buffers.blockCount = alloc->createBuffer(BlockCountLayout::size(blockCount),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                     memoryMapping);

        } else {
            buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferSrc,
                                                   memoryMapping);
            buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferSrc,
                                                memoryMapping);

            buffers.indices = alloc->createBuffer(IndicesLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                  memoryMapping);
            buffers.blockCount = alloc->createBuffer(BlockCountLayout::size(blockCount),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferDst,
                                                     memoryMapping);
        }
        return buffers;
    }
};

struct PartitionBlockScanConfig {
    const glsl::uint workgroupSize;
    const glsl::uint rows;
    const glsl::uint sequentialScanLength;
    const BlockScanVariant variant;

    constexpr PartitionBlockScanConfig(glsl::uint workgroupSize,
                                       glsl::uint rows,
                                       glsl::uint sequentialScanLength,
                                       BlockScanVariant variant)
        : workgroupSize(workgroupSize), rows(rows), sequentialScanLength(sequentialScanLength),
          variant(variant) {}

    inline constexpr glsl::uint blockSize() const {
        return workgroupSize * rows * sequentialScanLength;
    }
};

class PartitionBlockScan {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = PartitionBlockScanBuffers;

    explicit PartitionBlockScan(const merian::ContextHandle& context,
        const merian::ShaderCompilerHandle& shaderCompiler,
                                PartitionBlockScanConfig config)
        : m_blockSize(config.blockSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // pivot (condition)
                .add_binding_storage_buffer() // indices
                .add_binding_storage_buffer() // block counts
                .build_push_descriptor_layout(context);

        const std::string shaderPath =
            "src/wrs/algorithm/partition/block_wise/block_scan/shader.comp";

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
        } else {
            throw std::runtime_error("Unsupported PartitionBlockScan variant");
        }
        if ((config.variant & BlockScanVariant::INCLUSIVE) == BlockScanVariant::INCLUSIVE) {
            throw std::runtime_error("Unsupported PartitionBlockScan variant");
        }

        if ((config.variant & BlockScanVariant::STRIDED) == BlockScanVariant::STRIDED) {
            if ((config.variant & BlockScanVariant::RAKING) == BlockScanVariant::RAKING) {
                throw std::runtime_error("Unsupported variant");
            }
            defines["STRIDED"];
        }

        defines["USE_FLOAT"]; // NOTE only support float partitions

        const merian::ShaderModuleHandle shader =
            shaderCompiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute,
                {"src/wrs/algorithm/include"}, defines);

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
        specInfoBuilder.add_entry(config.sequentialScanLength);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, glsl::uint N) {

        cmd->bind(m_pipeline);
        if (buffers.blockCount != nullptr) {
            cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot, buffers.indices,
                                            buffers.blockCount);
        } else {
            cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot, buffers.indices);
        }
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
