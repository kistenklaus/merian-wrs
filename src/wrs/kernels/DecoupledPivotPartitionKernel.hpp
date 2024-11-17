#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
namespace wrs {

class DecoupledPivotPartitionKernel {
    using weight_t = float;

  public:
    DecoupledPivotPartitionKernel(const merian::ContextHandle& context,
                                  uint32_t workgroup_size,
                                  uint32_t rows) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, "src/wrs/kernels/decoupled_pivot_partition.comp",
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
                  uint32_t workgroup_count,
                  merian::BufferHandle in_elementBuffer,
                  uint32_t elementCount,
                  merian::BufferHandle in_pivotBuffer,
                  merian::BufferHandle out_partition,
                  merian::BufferHandle partitionDescriptorBuffer) {
        m_pipeline->bind(cmd);
        /* m_pipeline->push_descriptor_set(cmd, in_elementBuffer, out_partition, */
        /*                                 partitionDescriptorBuffer, in_pivotBuffer); */
        auto elementBufferInfo = in_elementBuffer->get_descriptor_info();
        auto partitionInfo = out_partition->get_descriptor_info();
        auto partitionDescriptorInfo = partitionDescriptorBuffer->get_descriptor_info();
        auto pivotBuffer = vk::DescriptorBufferInfo{*in_pivotBuffer, 0, sizeof(float)};

        m_pipeline->push_descriptor_set(
            cmd,
            {
                vk::WriteDescriptorSet{
                    {}, 0, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &elementBufferInfo, {}},
                vk::WriteDescriptorSet{
                    {}, 1, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &partitionInfo, {}},
                vk::WriteDescriptorSet{{},
                                       2,
                                       0,
                                       1,
                                       vk::DescriptorType::eStorageBuffer,
                                       {},
                                       &partitionDescriptorInfo,
                                       {}},
                vk::WriteDescriptorSet{
                    {}, 3, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &pivotBuffer, {}},
            });

        m_pipeline->push_constant(cmd, elementCount);
        cmd.dispatch(workgroup_count, 1, 1);
    }

    vk::DeviceSize partitionDescriptorBufferSize(uint32_t workgroupCount) const {
        return sizeof(unsigned int) +
               (2 * sizeof(weight_t) + sizeof(unsigned int)) * workgroupCount;
    }

    uint32_t partitionSize() const {
        return m_partitionSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    uint32_t m_partitionSize;
};

} // namespace wrs
