#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include <iostream>
namespace wrs {

class WorkgroupReduceKernel {

  public:
    using weight_t = float;
    WorkgroupReduceKernel(const merian::ContextHandle& context,
                          uint32_t workgroup_size,
                          uint32_t rows) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, "src/wrs/kernels/workgroup_reduce_kernel.comp",
                vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(
            workgroup_size,
            context->physical_device.physical_device_subgroup_properties.subgroupSize, rows);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
        m_partitionSize = workgroup_size * rows;
    }

    void dispatch(vk::CommandBuffer cmd,
                  uint32_t elementCount,
                  merian::BufferHandle in_weights,
                  merian::BufferHandle out_workgroupReduction) {
        m_pipeline->bind(cmd);

        m_pipeline->push_descriptor_set(cmd, in_weights, out_workgroupReduction);

        m_pipeline->push_constant(cmd, elementCount);

        // ceil int div
        cmd.dispatch(expectedWorkgroupCount(elementCount), 1, 1);
    }

    uint32_t getPartitionSize() const {
        return m_partitionSize;
    }

    uint32_t expectedWorkgroupCount(uint32_t elementCount) const {
        return (elementCount + m_partitionSize - 1) / m_partitionSize;
    }

    uint32_t expectedResultCount(uint32_t elementCount) {
        return elementCount / m_partitionSize + 1;
    }

  private:
    merian::PipelineHandle m_pipeline;
    uint32_t m_partitionSize;
};

} // namespace wrs
