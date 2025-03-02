#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
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
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

template <typename T = float> struct BlockScanBuffers {
    using Self = BlockScanBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    using ElementsView = host::layout::BufferView<ElementsLayout>;

    merian::BufferHandle reductions; // optional buffer if nullptr no reductions are written!
    using ReductionsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    using ReductionsView = host::layout::BufferView<ReductionsLayout>;

    merian::BufferHandle prefixSum;
    using PrefixSumLayout = host::layout::ArrayLayout<T, storageQualifier>;
    using PrefixSumView = host::layout::BufferView<PrefixSumLayout>;

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
            buffers.reductions = alloc->createBuffer(ReductionsLayout::size(blockCount),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                     memoryMapping);
            buffers.prefixSum = alloc->createBuffer(PrefixSumLayout::size(N),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferSrc,
                                                    memoryMapping);

        } else {
            buffers.elements = alloc->createBuffer(
                ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.reductions =
                alloc->createBuffer(ReductionsLayout::size(blockCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.prefixSum = alloc->createBuffer(
                PrefixSumLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

struct BlockScanConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;
    const BlockScanVariant variant;
    const host::glsl::uint sequentialScanLength;
    const bool writeBlockReductions;

    constexpr BlockScanConfig()
        : workgroupSize(512), rows(8), variant(BlockScanVariant::RAKING), sequentialScanLength(1),
          writeBlockReductions(true) {}
    explicit constexpr BlockScanConfig(host::glsl::uint workgroupSize,
                                       host::glsl::uint rows,
                                       BlockScanVariant variant,
                                       host::glsl::uint sequentialScanLength,
                                       bool writeBlockReductions)
        : workgroupSize(workgroupSize), rows(rows), variant(variant),
          sequentialScanLength(sequentialScanLength), writeBlockReductions(writeBlockReductions) {}

    constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows * sequentialScanLength;
    }
};

template <typename T = float> class BlockScan {
    struct PushConstants {
        host::glsl::uint N;
    };

  public:
    using Buffers = BlockScanBuffers<T>;
    using Config = BlockScanConfig;

    explicit BlockScan(const merian::ContextHandle& context,
                       const merian::ShaderCompilerHandle& shaderCompiler,
                       BlockScanConfig config = {})
        : m_blockSize(config.blockSize()), m_writeReductions(config.writeBlockReductions) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // reductions
                .add_binding_storage_buffer() // prefix sum
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/prefix_sum/block_scan/shader.comp";

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
            const auto it = defines.find("EXCLUSIVE");
            if (it != defines.end()) {
                defines.erase(it);
            }
        }

        if ((config.variant & BlockScanVariant::STRIDED) == BlockScanVariant::STRIDED) {
            if ((config.variant & BlockScanVariant::RAKING) == BlockScanVariant::RAKING) {
                throw std::runtime_error("Unsupported variant");
            }
            defines["STRIDED"];
        }

        if constexpr (std::is_same_v<T, host::glsl::f32>) {
            defines["USE_FLOAT"];
        } else if constexpr (std::is_same_v<T, host::glsl::uint>) {
            defines["USE_UINT"];
        } else {
            throw std::runtime_error("unsupported type for BlockScans");
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
        specInfoBuilder.add_entry(static_cast<host::glsl::uint>(config.variant));
        specInfoBuilder.add_entry(config.sequentialScanLength);
        specInfoBuilder.add_entry(
            static_cast<host::glsl::uint>(config.writeBlockReductions ? 1 : 0));
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void
    run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N) const {

        cmd->bind(m_pipeline);
        if (m_writeReductions) {
            assert(buffers.reductions != nullptr);
            cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.prefixSum,
                                     buffers.reductions);
        } else {
            cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.prefixSum);
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
    bool m_writeReductions;
};

} // namespace device
