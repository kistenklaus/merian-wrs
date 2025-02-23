#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/host/types/glsl.hpp"
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocations.hpp"

namespace device {

struct SampleAliasTableBuffers {
    using Self = SampleAliasTableBuffers;

    merian::BufferHandle aliasTable;
    merian::BufferHandle samples;
};

struct SampleAliasTableConfig {
    const host::glsl::uint cooperativeSampleSize;
    const host::glsl::uint workgroupSize;

    constexpr explicit SampleAliasTableConfig(host::glsl::uint cooperativeSampleSize,
                                              host::glsl::uint workgroupSize = 512)
        : cooperativeSampleSize(cooperativeSampleSize), workgroupSize(workgroupSize) {}
};

class SampleAliasTable {
    struct PushConstants {
        host::glsl::uint N;
        host::glsl::uint S;
        host::glsl::uint seed;
    };

  public:
    using Buffers = SampleAliasTableBuffers;
    using Config = SampleAliasTableConfig;

    explicit SampleAliasTable(const merian::ContextHandle& context,
                              const merian::ShaderCompilerHandle& shaderCompiler,
                              const SampleAliasTableConfig& config)
        : m_workgroupSize(config.workgroupSize) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // alias table
                .add_binding_storage_buffer() // samples
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/wrs/alias/sampling/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        specInfoBuilder.add_entry(config.cooperativeSampleSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             host::glsl::uint S,
             host::glsl::uint seed = 12345u) const {
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.aliasTable, buffers.samples);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N, .S = S, .seed = seed});
        const uint32_t workgroupCount = (S + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_workgroupSize;
};

} // namespace device
