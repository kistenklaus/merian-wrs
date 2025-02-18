#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct DecoupledPartitionBuffers {
    using Self = DecoupledPartitionBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                                    merian::MemoryMappingType memoryMapping) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {

        } else {

        }
        return buffers;
    }
};

class DecoupledPartition {
    struct PushConstants {
        glsl::uint X;
    };
  public:
    using Buffers = DecoupledPartitionBuffers;

    explicit DecoupledPartition(const merian::ContextHandle& context,
        const merian::ShaderCompilerHandle& shaderCompiler, glsl::uint workgroupSize) : m_workgroupSize(workgroupSize){

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/???";

        const merian::ShaderModuleHandle shader =
            shaderCompiler->find_compile_glsl_to_shadermodule(
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

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, TODO);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .X = TODO
        });
        const uint32_t workgroupCount = TODO;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

}
