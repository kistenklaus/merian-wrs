#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include <algorithm>
#include <iostream>
#include <vulkan/vulkan_handles.hpp>
namespace wrs {

class DecoupledPrefixSumAndAverageKernel {
  public:
    using weight_t = float;

  public:
    DecoupledPrefixSumAndAverageKernel(const merian::ContextHandle& context,
                             uint32_t workgroupSize,
                             uint32_t rows) {

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
                context, "src/wrs/kernels/decoupled_prefix_sum.comp",
                vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(
            workgroupSize, 
            context->physical_device.physical_device_subgroup_properties.subgroupSize,
            rows);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
        m_partitionSize = workgroupSize * rows;
    }

    void dispatch(vk::CommandBuffer cmd,
                  uint32_t workgroupCount,
                  uint32_t elementCount,
                  merian::BufferHandle in_elements,
                  merian::BufferHandle out_prefixSum,
                  merian::BufferHandle partitionDescriptorBuffer) {
        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd,in_elements, out_prefixSum, partitionDescriptorBuffer);
        m_pipeline->push_constant(cmd, elementCount);
        cmd.dispatch(workgroupCount, 1, 1);
    }

    vk::DeviceSize partitionDescriptorBufferSize(uint32_t workgroupCount) const {
        return sizeof(uint32_t) +
               (2 * sizeof(weight_t) + sizeof(uint32_t)) * workgroupCount;
    }

    uint32_t partitionSize() const {
        return m_partitionSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    uint32_t m_partitionSize;
};

} // namespace wrs
