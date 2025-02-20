#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct MeanSquaredErrorBuffers {
    using Self = MeanSquaredErrorBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle histogram;
    using HistogramLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using HistogramView = host::layout::BufferView<HistogramLayout>;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle mse;
    using MseLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using MseView = host::layout::BufferView<MseLayout>;

    static Self allocate([[maybe_unused]] const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping) {
        Self buffers;
        throw std::runtime_error("NOT IMPLEMENTED");
        if (memoryMapping == merian::MemoryMappingType::NONE) {

        } else {
        }
        return buffers;
    }
};

class MeanSquaredError {
    struct PushConstants {
        host::glsl::uint offset;
        float S;
        host::glsl::uint N;
        float totalWeight;
    };

  public:
    using Buffers = MeanSquaredErrorBuffers;

    explicit MeanSquaredError(const merian::ContextHandle& context,
                              const merian::ShaderCompilerHandle& shaderCompiler,
                              host::glsl::uint workgroupSize = 512,
                              host::glsl::uint rows = 8)
        : m_partitionSize(workgroupSize * rows) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // histogram
                .add_binding_storage_buffer() // weights
                .add_binding_storage_buffer() // rme
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/rmse/mse/shader.comp";

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
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint offset,
             float S,
             host::glsl::uint N,
             float totalWeight) {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.histogram, buffers.weights, buffers.mse);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .offset = offset,
                                                          .S = S,
                                                          .N = N,
                                                          .totalWeight = totalWeight,
                                                      });
        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_partitionSize;
};

} // namespace wrs
