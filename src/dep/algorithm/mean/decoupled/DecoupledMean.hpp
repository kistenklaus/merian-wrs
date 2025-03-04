#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : DecoupledMean.hpp
 *
 * Essentially a single dispatch prefix sum, which only writes back the last element
 * divided by N. As the single dispatch approach operates close to memory bandwidth limits
 * this does too. Compared to the AtomicMean approach this performs slightly worse, however
 * it does not require floating point atomics.
 */

#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/algorithm/mean/MeanAllocFlags.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
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

    static std::size_t partitionSize(std::size_t workgroupSize, std::size_t rows) {
        return workgroupSize * rows;
    }

    static DecoupledMeanBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                         std::size_t elementCount,
                                         std::size_t partitionSize,
                                         merian::MemoryMappingType memoryMapping,
                                         MeanAllocFlags allocFlags = MeanAllocFlags::ALLOC_ALL);

    static DecoupledMeanBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                         std::size_t elementCount,
                                         std::size_t workgroupSize,
                                         std::size_t rows,
                                         merian::MemoryMappingType memoryMapping,
                                         MeanAllocFlags allocFlags = MeanAllocFlags::ALLOC_ALL) {
        return allocate(alloc, elementCount, partitionSize(workgroupSize, rows), memoryMapping,
                        allocFlags);
    }
};

struct DecoupledMeanConfig {
    const glsl::uint workgroupSize;
    const glsl::uint rows;

    constexpr DecoupledMeanConfig() : workgroupSize(512), rows(4) {}
    explicit constexpr DecoupledMeanConfig(glsl::uint workgroupSize, glsl::uint rows)
        : workgroupSize(workgroupSize), rows(rows) {}

    constexpr glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

class DecoupledMean {

  public:
    using elem_t = float;
    using Buffers = DecoupledMeanBuffers;
    static constexpr uint32_t DEFAULT_WORKGROUP_SIZE = 512;
    static constexpr uint32_t DEFAULT_ROWS = 4;

    DecoupledMean(const merian::ContextHandle& context,
                  const merian::ShaderCompilerHandle& shaderCompiler,
                  DecoupledMeanConfig config)
        : m_blockSize(config.blockSize()) {
        constexpr bool stable = false;
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // mean
                .add_binding_storage_buffer() // decoupled states
                .build_push_descriptor_layout(context);
        std::string shaderPath;
        if (stable) {
            throw std::runtime_error("Not implemented yet");
            shaderPath = "src/wrs/algorithm/mean/decoupled/float_stable.comp";
        } else {
            shaderPath = "src/wrs/algorithm/mean/decoupled/float.comp";
        }
        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>() // size (N)
                .build_pipeline_layout();
        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(config.workgroupSize);
        specInfoBuilder.add_entry(context->physical_device.physical_device_subgroup_properties.subgroupSize);
        specInfoBuilder.add_entry(config.rows);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(merian::CommandBufferHandle cmd, const DecoupledMeanBuffers& buffers, uint32_t N) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.mean,
                                 buffers.decoupledStates);
        cmd->push_constant(m_pipeline, N);

        const uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline uint32_t getPartitionSize() const {
        return m_blockSize;
    }

  private:
    const uint32_t m_blockSize;
    merian::PipelineHandle m_pipeline;
};

}; // namespace wrs
