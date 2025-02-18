#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : Explode.hpp
 *
 * Horrible implementation of a explode operation known from SQL databases.
 * In our case this turns a histogram back into the individual samples in order.
 *
 * This implementation uses a decoupled prefix sum to find the indices to write the
 * samples. What makes this horrible, is that a single thread writes all samples for a
 * a given element, which leads to bad load balancing.
 */

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct ExplodeBuffers {
    using Self = ExplodeBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle outputSensitive;
    using OutputSensitiveLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using OutputSensitiveView = layout::BufferView<OutputSensitiveLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    merian::BufferHandle decoupledState;
    using _DecoupledStateLayout =
        wrs::layout::StructLayout<storageQualifier,
                                  layout::Attribute<glsl::uint, "aggregate">,
                                  layout::Attribute<glsl::uint, "prefix">,
                                  layout::Attribute<glsl::uint, "state">>;
    using _DecoupledStateArrayLayout =
        wrs::layout::ArrayLayout<_DecoupledStateLayout, storageQualifier>;
    using DecoupledStatesLayout =
        wrs::layout::StructLayout<storageQualifier,
                                  layout::Attribute<glsl::uint, "counter">,
                                  layout::Attribute<_DecoupledStateArrayLayout, "partitions">>;
    using DecoupledStatesView = layout::BufferView<DecoupledStatesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t S,
                         std::size_t partitionSize);
};

class Explode {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = ExplodeBuffers;

    explicit Explode(const merian::ContextHandle& context,
                     const merian::ShaderCompilerHandle shaderCompiler,
                     glsl::uint workgroupSize,
                     glsl::uint rows,
                     glsl::uint parallelLookbackDepth)
        : m_partitionSize(workgroupSize * rows) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // tree
                .add_binding_storage_buffer() // samples
                .add_binding_storage_buffer() // output senitive
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/hs/explode/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(workgroupSize);
        specInfoBuilder.add_entry(rows);
        specInfoBuilder.add_entry(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        specInfoBuilder.add_entry(parallelLookbackDepth);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, glsl::uint N) const {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.outputSensitive, buffers.samples,
                                 buffers.decoupledState);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});

        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint getPartitionSize() const {
        return m_partitionSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_partitionSize;
};

} // namespace wrs
