#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct SampleAliasTableBuffers {
    using Self = SampleAliasTableBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle aliasTable;
    using _AliasTableEntryLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<float, layout::StaticString("p")>,
                             layout::Attribute<int, layout::StaticString("a")>>;
    using AliasTableLayout = layout::ArrayLayout<_AliasTableEntryLayout, storageQualifier>;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t weightCount,
                         std::size_t sampleCount);
};

class SampleAliasTable {
    struct PushConstants {
        glsl::uint N;
        glsl::uint S;
        glsl::uint seed;
    };

  public:
    using Buffers = SampleAliasTableBuffers;

    explicit SampleAliasTable(const merian::ContextHandle& context,
                              const merian::ShaderCompilerHandle& shaderCompiler,
                              glsl::uint workgroupSize = 512) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // alias table
                .add_binding_storage_buffer() // samples
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/psa/sampling/shader.comp";

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
             glsl::uint N,
             glsl::uint S,
             glsl::uint seed = 12345u) {
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.aliasTable, buffers.samples);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N, .S = S, .seed = seed});
        const uint32_t workgroupCount = (S + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
