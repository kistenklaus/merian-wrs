#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <fmt/base.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
namespace wrs {

struct DecoupledMeanBuffers {
    using element_type = wrs::glsl::f32;
    // ELEMENTS
    merian::BufferHandle elements;
    static constexpr glsl::StorageQualifier elementStorageQualifier =
        glsl::StorageQualifier::std430;
    using ElementsLayout = layout::ArrayLayout<element_type, elementStorageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;
    static constexpr vk::BufferUsageFlags ELEMENT_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;

    // MEAN
    merian::BufferHandle mean;
    static constexpr vk::BufferUsageFlags MEAN_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    static constexpr glsl::StorageQualifier meanStorageQualifier = glsl::StorageQualifier::std430;
    using MeanLayout = layout::PrimitiveLayout<element_type, meanStorageQualifier>;
    using MeanView = layout::BufferView<MeanLayout>;

    // Decoupled State
    merian::BufferHandle decoupledStates;
    static constexpr glsl::StorageQualifier decoupledStatesStorageQualifier =
        glsl::StorageQualifier::std430;
    using _DecoupledStateLayout =
        wrs::layout::StructLayout<decoupledStatesStorageQualifier,
                                  layout::Attribute<element_type, "aggregate">,
                                  layout::Attribute<element_type, "prefix">,
                                  layout::Attribute<wrs::glsl::uint, "state">>;
    using _DecoupledStateArrayLayout =
        wrs::layout::ArrayLayout<_DecoupledStateLayout, decoupledStatesStorageQualifier>;
    using DecoupledStatesLayout =
        wrs::layout::StructLayout<decoupledStatesStorageQualifier,
                                  layout::Attribute<wrs::glsl::uint, "counter">,
                                  layout::Attribute<_DecoupledStateArrayLayout, "partitions">>;
    using DecoupledStatesView = layout::BufferView<DecoupledStatesLayout>;

    static constexpr vk::BufferUsageFlags DECOUPLED_STATE_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;

    static std::size_t partitionSize(std::size_t workgroupSize, std::size_t rows ) {
      return workgroupSize * rows;
    }

    static DecoupledMeanBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                         std::size_t elementCount,
                                         std::size_t partitionSize,
                                         merian::MemoryMappingType memoryMapping);

    static DecoupledMeanBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                         std::size_t elementCount,
                                         std::size_t workgroupSize,
                                         std::size_t rows,
                                         merian::MemoryMappingType memoryMapping) {
      return allocate(alloc, elementCount, partitionSize(workgroupSize, rows), memoryMapping);
    }
};

class DecoupledMean {

  public:
    using elem_t = float;
    using Buffers = DecoupledMeanBuffers;
    static constexpr uint32_t DEFAULT_WORKGROUP_SIZE = 512;
    static constexpr uint32_t DEFAULT_ROWS = 4;

    DecoupledMean(const merian::ContextHandle& context,
                  uint32_t workgroupSize = DEFAULT_WORKGROUP_SIZE,
                  uint32_t rows = DEFAULT_ROWS,
                  bool stable = false)
        : m_partitionSize(workgroupSize * rows) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // mean
                .add_binding_storage_buffer() // decoupled states
                .add_binding_storage_buffer() // decoupled aggregates
                .build_push_descriptor_layout(context);
        std::string shaderPath;
        if (stable) {
            throw std::runtime_error("Not implemented yet");
            shaderPath = "src/wrs/algorithm/mean/decoupled/float_stable.comp";
        } else {
            shaderPath = "src/wrs/algorithm/mean/decoupled/float.comp";
        }
        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>() // size
                .build_pipeline_layout();
        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(
            workgroupSize,
            context->physical_device.physical_device_subgroup_properties.subgroupSize, rows);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
        m_writes.resize(3);
        vk::WriteDescriptorSet& elements = m_writes[0];
        elements.setDstBinding(0);
        elements.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& mean = m_writes[1];
        mean.setDstBinding(1);
        mean.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& states = m_writes[2];
        states.setDstBinding(2);
        states.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    }

    void run(vk::CommandBuffer cmd, const DecoupledMeanBuffers& buffers, uint32_t N) {

        uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
                m_pipeline->bind(cmd);
        vk::DescriptorBufferInfo elementsDesc = buffers.elements->get_descriptor_info();
        m_writes[0].setBufferInfo(elementsDesc);
        vk::DescriptorBufferInfo meanDesc = buffers.mean->get_descriptor_info();
        m_writes[1].setBufferInfo(meanDesc);
        vk::DescriptorBufferInfo statesDesc = buffers.decoupledStates->get_descriptor_info();
        m_writes[2].setBufferInfo(statesDesc);

        m_pipeline->push_descriptor_set(cmd, m_writes);
        m_pipeline->push_constant(cmd, N);

        cmd.dispatch(workgroupCount, 1, 1);
    }

    inline uint32_t getPartitionSize() const {
      return m_partitionSize;
    }

  private:
    const uint32_t m_partitionSize;
    merian::PipelineHandle m_pipeline;
    std::vector<vk::WriteDescriptorSet> m_writes;
};

}; // namespace wrs
