#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace wrs {

class FinalAggAvgKernel {
  public:
    FinalAggAvgKernel(const merian::ContextHandle& context) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, "src/wrs/kernels/final_agg_avg_kernel.comp",
                vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>()
                .build_pipeline_layout();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader);
    }

    void dispatch(vk::CommandBuffer cmd,
                  merian::BufferHandle inout_meta,
                  uint32_t N) {
        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, inout_meta);
        m_pipeline->push_constant(cmd, N);
        cmd.dispatch(1, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
};
} // namespace wrs
