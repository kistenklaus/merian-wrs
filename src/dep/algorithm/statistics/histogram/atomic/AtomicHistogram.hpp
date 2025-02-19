#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : AtomicHistogram
 *
 * Horrible implementation of a histogram using atomic couters in every single invocation,
 * but good enough for evaluation.
 */

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct AtomicHistogramBuffers {
    using Self = AtomicHistogramBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    merian::BufferHandle histogram;
    using HistogramLayout = layout::ArrayLayout<glsl::uint64, storageQualifier>;
    using HistogramView = layout::BufferView<HistogramLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N,
                         glsl::uint S) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.samples = alloc->createBuffer(
                SamplesLayout::size(S), vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            buffers.histogram = alloc->createBuffer(HistogramLayout::size(N),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferSrc,
                                                    memoryMapping);

        } else {
            buffers.samples = nullptr;
            buffers.histogram = alloc->createBuffer(
                HistogramLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

class AtomicHistogram {
    struct PushConstants {
        glsl::uint offset;
        glsl::uint count;
    };

  public:
    using Buffers = AtomicHistogramBuffers;

    explicit AtomicHistogram(const merian::ContextHandle& context,
                             merian::ShaderCompilerHandle shaderCompiler,
                             glsl::uint workgroupSize = 512)
        : m_workgroupSize(workgroupSize) {
        assert(context != nullptr);
        assert(shaderCompiler != nullptr);

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // samples
                .add_binding_storage_buffer() // histogram
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/histogram/atomic/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
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

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             glsl::uint offset,
             glsl::uint count) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.samples, buffers.histogram);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .offset = offset,
                                                          .count = count,
                                                      });

        const uint32_t workgroupCount = (count + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
