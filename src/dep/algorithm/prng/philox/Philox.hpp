#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
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

struct PhiloxBuffers {
    using Self = PhiloxBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint sampleCount) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.samples = alloc->createBuffer(SamplesLayout::size(sampleCount),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  merian::MemoryMappingType::NONE);
        } else {
            buffers.samples =
                alloc->createBuffer(SamplesLayout::size(sampleCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

class PhiloxConfig {
  public:
    glsl::uint workgroupSize;

    constexpr PhiloxConfig() : workgroupSize(512) {}
    explicit constexpr PhiloxConfig(glsl::uint workgroupSize) : workgroupSize(workgroupSize) {}
};

class Philox {
    struct PushConstants {
        glsl::uint seed;
        glsl::uint N;
    };

  public:
    using Buffers = PhiloxBuffers;

    explicit Philox(const merian::ContextHandle& context,
                    const merian::ShaderCompilerHandle& shaderCompiler,
                    PhiloxConfig config = {})
        : m_workgroupSize(config.workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/prng/philox/philox.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(config.workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             glsl::uint sampleCount,
             glsl::uint N,
             glsl::uint seed = 12345u) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.samples);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .seed = seed,
                                                          .N = N,
                                                      });
        const uint32_t workgroupCount = (sampleCount + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
