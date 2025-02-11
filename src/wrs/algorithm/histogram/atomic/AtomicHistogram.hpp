#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
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

    explicit AtomicHistogram(const merian::ContextHandle& context, glsl::uint workgroupSize = 512)
        : m_workgroupSize(workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // samples
                .add_binding_storage_buffer() // histogram
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/histogram/atomic/shader.comp";
        

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

    void
    run(const vk::CommandBuffer& cmd, const Buffers& buffers, glsl::uint offset, glsl::uint count) {
        
        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.samples, buffers.histogram);
        m_pipeline->push_constant<PushConstants>(cmd,
                                                 PushConstants{.offset = offset, .count = count});
        const uint32_t workgroupCount = (count + m_workgroupSize - 1) / m_workgroupSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
