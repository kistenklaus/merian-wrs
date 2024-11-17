#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include <fmt/base.h>

namespace wrs::baseline {

class SplitKernel {
  public:
    struct PushConstants {
      uint weightCount;
      uint splitCount;
    };
    using weight_t = float;

    SplitKernel(const merian::ContextHandle& context) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, "src/wrs/alias_table/baseline/kernels/split.comp",
                vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void dispatch(vk::CommandBuffer cmd,
                  merian::BufferHandle in_weights,
                  uint32_t weightCount,
                  merian::BufferHandle in_avgPrefixSum,
                  merian::BufferHandle in_lightHeavy,
                  merian::BufferHandle in_lightHeavyPrefix,
                  merian::BufferHandle out_splitInfo,
                  uint32_t splitCount) {
        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, in_weights, in_avgPrefixSum, in_lightHeavy,
                                        in_lightHeavyPrefix, out_splitInfo);
        PushConstants constants{weightCount, splitCount};
        fmt::println("dispatch with {}, {}", weightCount, splitCount);
        m_pipeline->push_constant(cmd, constants);
        cmd.dispatch(splitCount, 1, 1);
    }

    vk::DeviceSize splitDescriptorSize() {
      return sizeof(unsigned int) + sizeof(unsigned int) + sizeof(float);
    }

  private:
    merian::PipelineHandle m_pipeline;
};

} // namespace wrs::baseline
