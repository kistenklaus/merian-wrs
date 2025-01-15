#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/algorithm/hs/HSTRepr.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <fmt/base.h>
#include <memory>
#include <ranges>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct HSTSamplingBuffers {
    using Self = HSTSamplingBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle hst;
    using HstLayout = layout::ArrayLayout<float, storageQualifier>;
    using HstView = layout::BufferView<HstLayout>;

    using TreeLayout = layout::ArrayLayout<float, storageQualifier>;
    using TreeView = layout::BufferView<TreeLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N) {

        hst::HSTRepr repr{N};
        std::size_t entries = repr.size();

        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.hst = alloc->createBuffer(HstLayout::size(entries),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
            buffers.samples = alloc->createBuffer(SamplesLayout::size(entries + 1),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                  memoryMapping);
        } else {
            buffers.hst = alloc->createBuffer(HstLayout::size(entries),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.samples = alloc->createBuffer(SamplesLayout::size(entries + 1),
                                                  vk::BufferUsageFlagBits::eTransferDst |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);
        }
        return buffers;
    }
};

class HSTSampling {
    struct PushConstants {
        glsl::uint child_offset;
        glsl::uint parent_offset;
        glsl::uint num_invoc;
    };

  public:
    using Buffers = HSTSamplingBuffers;

    explicit HSTSampling(const merian::ContextHandle& context, glsl::uint workgroupSize)
        : m_workgroupSize(workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // hst
                .add_binding_storage_buffer() // samples
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/hs/sampling/shader.comp";

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

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, std::size_t N) {

        m_pipeline->bind(cmd);
        hst::HSTRepr repr{N};

        m_pipeline->push_descriptor_set(cmd, buffers.hst, buffers.samples);

        /* fmt::println("hst_offset = {}", repr.get().back().parentOffset); */

        glsl::uint parentOffset = static_cast<glsl::uint>(repr.size());
        glsl::uint invoc = 1;
        for (const auto& level : repr.get() | std::views::reverse) {
            /* fmt::println("child_offset = {}, parent_offset = {}, invoc={}", level.parentOffset, */
            /*              parentOffset, invoc); */

            m_pipeline->push_constant<PushConstants>(cmd, PushConstants{
                                                              .child_offset = level.parentOffset,
                                                              .parent_offset = parentOffset,
                                                              .num_invoc = invoc,
                                                          });
            const glsl::uint workgroupCount = (invoc + m_workgroupSize - 1) / m_workgroupSize;
            cmd.dispatch(workgroupCount, 1, 1);

            parentOffset = level.parentOffset;
            invoc = level.numParents;

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader, {}, {},
                                buffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                vk::AccessFlagBits::eShaderRead),
                                {});
        }

        m_pipeline->push_constant<PushConstants>(cmd,
                                                 PushConstants{
                                                     .child_offset = repr.get().front().childOffset,
                                                     .parent_offset = parentOffset,
                                                     .num_invoc = invoc,
                                                 });

        const glsl::uint workgroupCount = (invoc + m_workgroupSize - 1) / m_workgroupSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
