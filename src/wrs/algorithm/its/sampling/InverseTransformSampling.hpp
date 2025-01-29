#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <memory>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct InverseTransformSamplingBuffers {
    using Self = InverseTransformSamplingBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle cmf; // cummulative mass function
    using CMFLayout = layout::ArrayLayout<float, storageQualifier>;
    using CMFView = layout::BufferView<CMFLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t cmfSize,
                         std::size_t sampleCount) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.cmf = alloc->createBuffer(CMFLayout::size(cmfSize),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              merian::MemoryMappingType::NONE);
            buffers.samples = alloc->createBuffer(SamplesLayout::size(sampleCount),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  merian::MemoryMappingType::NONE);
        } else {
            buffers.cmf = alloc->createBuffer(CMFLayout::size(cmfSize),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.samples =
                alloc->createBuffer(SamplesLayout::size(sampleCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

class InverseTransformSamplingConfig {
  public:
    glsl::uint workgroupSize;
    glsl::uint cooperativeSamplingSize;

    constexpr InverseTransformSamplingConfig() : workgroupSize(512), cooperativeSamplingSize(4096) {}
    explicit constexpr InverseTransformSamplingConfig(glsl::uint workgroupSize,
                                            glsl::uint cooperativeSamplingSize)
        : workgroupSize(workgroupSize), cooperativeSamplingSize(cooperativeSamplingSize) {}
};

class InverseTransformSampling {
    struct PushConstants {
        glsl::uint N; // cmf size
        glsl::uint S; // sample count
        glsl::uint seed;
    };

  public:
    using Buffers = InverseTransformSamplingBuffers;

    explicit InverseTransformSampling(const merian::ContextHandle& context,
                                      InverseTransformSamplingConfig config = {})
        : m_workgroupSize(config.workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/its/sampling/shader.comp";

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
        specInfoBuilder.add_entry(config.cooperativeSamplingSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd,
             const Buffers& buffers,
             glsl::uint N,
             glsl::uint S,
             glsl::uint seed = 12345u) {

        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.cmf, buffers.samples);
        m_pipeline->push_constant<PushConstants>(cmd, PushConstants{
                                                          .N = N,
                                                          .S = S,
                                                          .seed = seed,
                                                      });
        const uint32_t workgroupCount = (S + m_workgroupSize - 1) / m_workgroupSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
