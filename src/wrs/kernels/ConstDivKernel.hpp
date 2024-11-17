#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace wrs {

class ConstDivKernel {
  public:
    struct PushConstants {
        unsigned int size;
        float div;
    };
    ConstDivKernel(const merian::ContextHandle& context, uint32_t workgroupSize) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, "src/wrs/kernels/const_div_kernel.comp",
                vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(
            workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void dispatch(vk::CommandBuffer cmd,
                  uint32_t workgroupCount,
                  uint32_t elementCount,
                  merian::BufferHandle src,
                  merian::BufferHandle dst,
                  float div) {
        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, src, dst);
        m_pipeline->push_constant(cmd, PushConstants{elementCount, div});
        cmd.dispatch(workgroupCount, 1, 1);
    }

    void dispatch(vk::CommandBuffer cmd,
                  uint32_t workgroupCount,
                  uint32_t elementCount,
                  vk::WriteDescriptorSet src,
                  vk::WriteDescriptorSet dst,
                  float div) {
        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, {src, dst});
        m_pipeline->push_constant(cmd, PushConstants{elementCount, div});
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
};
} // namespace wrs
