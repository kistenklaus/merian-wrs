#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/algorithm/hs/HSTRepr.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct HSTCBuffers {
    using Self = HSTCBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle tree;
    using TreeLayout = layout::ArrayLayout<float, storageQualifier>;
    using TreeView = layout::BufferView<TreeLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.tree = alloc->createBuffer(TreeLayout::size(2 * N - 2),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst |
                                                   vk::BufferUsageFlagBits::eTransferSrc,
                                               merian::MemoryMappingType::NONE);
        } else {
            buffers.tree = alloc->createBuffer(TreeLayout::size(2 * N - 2),
                                               vk::BufferUsageFlagBits::eTransferSrc |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               memoryMapping);
        }
        return buffers;
    }
};

class HSTC {
    struct PushConstants {
        glsl::uint dst_offset;
        glsl::uint src_offset;
        glsl::uint num_invoc;
    };

  public:
    using Buffers = HSTCBuffers;

    explicit HSTC(const merian::ContextHandle& context, glsl::uint workgroupSize)
        : m_workgroupSize(workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // tree
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/hs/hstc/shader.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, const glsl::uint N) {

        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.tree);

        hst::HSTRepr repr{N};

        bool first = false;
        for (const auto& level : repr.get()) {

            if (!first) {
                cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader, {}, {},
                                    buffers.tree->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead),
                                    {});
            }
            first = false;

            glsl::uint dstOffset = level.parentOffset + (level.overlap ? 1u : 0u);
            glsl::uint srcOffset = level.childOffset;
            glsl::uint numInvoc = level.numParents - (level.overlap ? 1u : 0u);
            m_pipeline->push_constant<PushConstants>(cmd, PushConstants{
                                                              .dst_offset = dstOffset,
                                                              .src_offset = srcOffset,
                                                              .num_invoc = numInvoc,
                                                          });
            const uint32_t workgroupCount = (numInvoc + m_workgroupSize - 1) / m_workgroupSize;
            cmd.dispatch(workgroupCount, 1, 1);
        }
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
