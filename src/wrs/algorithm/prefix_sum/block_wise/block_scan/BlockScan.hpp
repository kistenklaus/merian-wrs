#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <type_traits>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct BlockScanBuffers {
    using Self = BlockScanBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    merian::BufferHandle reductions; // optional buffer if nullptr no reductions are written!
    using ReductionsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ReductionsView = layout::BufferView<ReductionsLayout>;

    merian::BufferHandle prefixSum;
    using PrefixSumLayout = layout::ArrayLayout<float, storageQualifier>;
    using PrefixSumView = layout::BufferView<PrefixSumLayout>;

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

enum class BlockScanVariant : glsl::uint {
    WORKGROUP_HILLIS_STEEL = 1,
    WORKGROUP_SUBGROUP_SCAN = 1,
    SUBGROUP_INTRINSIC = 0x100,
    SUBGROUP_HILLIS_STEEL = 0x200,
};

inline constexpr BlockScanVariant operator|(BlockScanVariant a, BlockScanVariant b) noexcept {
    using Base = std::underlying_type_t<BlockScanVariant>;
    return static_cast<BlockScanVariant>(static_cast<Base>(a) | static_cast<Base>(b));
}

struct BlockScanConfig {
    const glsl::uint workgroupSize;
    const glsl::uint rows;
    const BlockScanVariant variant;
    const glsl::uint sequentialScanLength;
    const bool writeBlockReductions;

    constexpr BlockScanConfig()
        : workgroupSize(512), rows(8), variant(BlockScanVariant::SUBGROUP_INTRINSIC),
          sequentialScanLength(1), writeBlockReductions(true) {}
    explicit constexpr BlockScanConfig(glsl::uint workgroupSize,
                                       glsl::uint rows,
                                       BlockScanVariant variant,
                                       glsl::uint sequentialScanLength,
                                       bool writeBlockReductions)
        : workgroupSize(workgroupSize), rows(rows), variant(variant),
          sequentialScanLength(sequentialScanLength), writeBlockReductions(writeBlockReductions) {}

    constexpr glsl::uint blockSize() const {
        return workgroupSize * rows * 2 * sequentialScanLength;
    }
};

class BlockScan {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = BlockScanBuffers;

    explicit BlockScan(const merian::ContextHandle& context, BlockScanConfig config = {})
        : m_blockSize(config.blockSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // reductions
                .add_binding_storage_buffer() // prefix sum
                .build_push_descriptor_layout(context);

        const std::string shaderPath =
            "src/wrs/algorithm/prefix_sum/block_wise/block_scan/shader.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute,
                {"src/wrs/algorithm/include/"});

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
        specInfoBuilder.add_entry(static_cast<glsl::uint>(config.variant));
        specInfoBuilder.add_entry(config.sequentialScanLength);
        specInfoBuilder.add_entry(static_cast<glsl::uint>(config.writeBlockReductions ? 1 : 0));
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, glsl::uint N) {

        m_pipeline->bind(cmd);
        if (buffers.reductions == nullptr) {
            m_pipeline->push_descriptor_set(cmd, buffers.elements, buffers.prefixSum);
        } else {
            m_pipeline->push_descriptor_set(cmd, buffers.elements, buffers.prefixSum,
                                            buffers.reductions);
        }
        m_pipeline->push_constant<PushConstants>(cmd, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;
        fmt::println("DISPATCH: {}   -- partitionSize: {}", workgroupCount, m_blockSize);
        cmd.dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint blockSize() const {
        return m_blockSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_blockSize;
};

} // namespace wrs
