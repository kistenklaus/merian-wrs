#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct PartitionBlockScanBuffers {
    using Self = PartitionBlockScanBuffers;
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

    merian::BufferHandle blockCount;
    using BlockCountLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using BlockCountView = host::layout::BufferView<BlockCountLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N,
                         host::glsl::uint blockCount) {
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
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;
    const host::glsl::uint sequentialScanLength;
    const BlockScanVariant variant;

    constexpr PartitionBlockScanConfig(host::glsl::uint workgroupSize,
                                       host::glsl::uint rows,
                                       host::glsl::uint sequentialScanLength,
                                       BlockScanVariant variant)
        : workgroupSize(workgroupSize), rows(rows), sequentialScanLength(sequentialScanLength),
          variant(variant) {}

    inline constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows * sequentialScanLength;
    }
};

class PartitionBlockScan {
    struct PushConstants {
        host::glsl::uint N;
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
            "src/device/partition/block_wise/block_scan/shader.comp";

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

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {"src/device/common"},
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
        specInfoBuilder.add_entry(config.sequentialScanLength);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N) {

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

    inline host::glsl::uint blockSize() const {
        return m_blockSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_blockSize;
};

} // namespace device
