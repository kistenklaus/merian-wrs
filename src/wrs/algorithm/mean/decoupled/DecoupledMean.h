#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include <concepts>
#include <fmt/base.h>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
namespace wrs {

class DecoupledMeanBuffers {
  public:
    merian::BufferHandle elements;
    static constexpr vk::BufferUsageFlags ELEMENT_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;

    merian::BufferHandle mean;
    static constexpr vk::BufferUsageFlags MEAN_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;

    merian::BufferHandle decoupledStates;
    static constexpr vk::BufferUsageFlags DECOUPLED_STATE_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    static constexpr vk::DeviceSize minDecoupledStateSize(uint32_t N, uint32_t partitionSize) {
        const uint32_t workgroupCount = (N + partitionSize - 1) / partitionSize;
        const vk::DeviceSize stateSize = sizeof(uint32_t) + sizeof(float) * 2;
        return workgroupCount * stateSize + // partitionStates states
               sizeof(uint32_t);            // atomic partition counter
    }
    static constexpr vk::DeviceSize
    minDecoupledStateSize(uint32_t N, uint32_t workgroupSize, uint32_t rows) {
        return minDecoupledStateSize(N, workgroupSize * rows);
    }

    merian::BufferHandle decoupledAggregates;
    static constexpr vk::BufferUsageFlags DECOUPLED_AGGREGATES_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    static constexpr vk::DeviceSize
    minDecoupledAggregatesSize(uint32_t N, uint32_t partitionSize, size_t sizeOfElement) {
        const uint32_t workgroupCount = (N + partitionSize - 1) / partitionSize;
        vk::DeviceSize elementSize = 2 * sizeOfElement;
        return workgroupCount * elementSize * 2;
    }
    static constexpr vk::DeviceSize minDecoupledAggregatesSize(uint32_t N,
                                                               uint32_t workgroupSize,
                                                               uint32_t rows,
                                                               size_t sizeOfElement) {
        return minDecoupledAggregatesSize(N, workgroupSize * rows, sizeOfElement);
    }
};

template <typename T = float> class DecoupledMean {
    static_assert(std::same_as<T, float>);

  private:
#ifdef NDEBUG
    static constexpr bool CHECK_PARAMETERS = false;
#else
    static constexpr bool CHECK_PARAMETERS = true;
#endif
  public:
    using elem_t = T;
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
        specInfoBuilder.add_entry(workgroupSize, context->physical_device.physical_device_subgroup_properties.subgroupSize, rows);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
        m_writes.resize(4);
        vk::WriteDescriptorSet& elements = m_writes[0];
        elements.setDstBinding(0);
        elements.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& mean = m_writes[1];
        mean.setDstBinding(1);
        mean.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& states = m_writes[2];
        states.setDstBinding(2);
        states.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& agg = m_writes[3];
        agg.setDstBinding(3);
        agg.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    }

    void run(vk::CommandBuffer cmd, const DecoupledMeanBuffers& buffers, uint32_t N) {
        if constexpr (CHECK_PARAMETERS) {
            // CHECK NULL POINTERS
            if (cmd == VK_NULL_HANDLE) {
                throw std::runtime_error("cmd is VK_NULL_HANDLE");
            }
            if (buffers.elements->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.elements is VK_NULL_HANDLE");
            }
            if (buffers.mean->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.mean is VK_NULL_HANDLE");
            }
            if (buffers.decoupledStates->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.decoupledStates is VK_NULL_HANDLE");
            }
            if (buffers.decoupledAggregates->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.decoupledAggregates is VK_NULL_HANDLE");
            }
            // CHECK BUFFER SIZES
            if (buffers.elements->get_size() < sizeof(elem_t) * N) {
                throw std::runtime_error("buffers.elements is to small!");
            }
            if (buffers.mean->get_size() < sizeof(elem_t)) {
                throw std::runtime_error("buffers.mean is to small!");
            }
            if (buffers.decoupledStates->get_size() <
                DecoupledMeanBuffers::minDecoupledStateSize(N, m_partitionSize)) {
                throw std::runtime_error("buffers.decoupledStates is to small!");
            }
            if (buffers.decoupledAggregates->get_size() <
                DecoupledMeanBuffers::minDecoupledAggregatesSize(N, m_partitionSize, sizeof(elem_t))) {
                throw std::runtime_error("buffers.decoupledAggregates is to small!");
            }
        }
        m_pipeline->bind(cmd);
        vk::DescriptorBufferInfo elementsDesc = buffers.elements->get_descriptor_info();
        m_writes[0].setBufferInfo(elementsDesc);
        vk::DescriptorBufferInfo meanDesc = buffers.mean->get_descriptor_info();
        m_writes[1].setBufferInfo(meanDesc);
        vk::DescriptorBufferInfo statesDesc = buffers.decoupledStates->get_descriptor_info();
        m_writes[2].setBufferInfo(statesDesc);
        vk::DescriptorBufferInfo aggDesc = buffers.decoupledAggregates->get_descriptor_info();
        m_writes[3].setBufferInfo(aggDesc);

        m_pipeline->push_descriptor_set(cmd, m_writes);
        m_pipeline->push_constant(cmd, N);

        uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    const uint32_t m_partitionSize;
    merian::PipelineHandle m_pipeline;
    std::vector<vk::WriteDescriptorSet> m_writes;
};

}; // namespace wrs
