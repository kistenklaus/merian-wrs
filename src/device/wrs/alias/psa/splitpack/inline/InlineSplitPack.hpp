#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

namespace device {

struct InlineSplitPackBuffers {
    using Self = InlineSplitPackBuffers;

    merian::BufferHandle weights;

    merian::BufferHandle partitionIndices;

    merian::BufferHandle partitionPrefix;

    merian::BufferHandle heavyCount;

    merian::BufferHandle mean;

    merian::BufferHandle aliasTable;

    merian::BufferHandle partitionElements; // optional
};

struct InlineSplitPackConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint splitSize;

    constexpr explicit InlineSplitPackConfig(host::glsl::uint splitSize,
                                             const host::glsl::uint workgroupSize = 512)
        : workgroupSize(workgroupSize), splitSize(splitSize) {}
};

class InlineSplitPack {
    struct PushConstants {
        host::glsl::uint K;
        host::glsl::uint N;
    };

  public:
    using Buffers = InlineSplitPackBuffers;
    using Config = InlineSplitPackConfig;

    explicit InlineSplitPack(const merian::ContextHandle& context,
                             const merian::ShaderCompilerHandle& shaderCompiler,
                             const Config config,
                             bool usePartitionElements = false)
        : m_workgroupSize(config.workgroupSize), m_splitSize(config.splitSize),
          m_usePartitionElements(usePartitionElements) {

        auto setBuilder = merian::DescriptorSetLayoutBuilder()
                              .add_binding_storage_buffer()  // weights
                              .add_binding_storage_buffer()  // partition indices
                              .add_binding_storage_buffer()  // partition prefix
                              .add_binding_storage_buffer()  // heavy count
                              .add_binding_storage_buffer()  // mean
                              .add_binding_storage_buffer(); // alias table

        if (m_usePartitionElements) {
            setBuilder.add_binding_storage_buffer(); // partition elements
        }
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            setBuilder.build_push_descriptor_layout(context);

        const std::string shaderPath =
            "src/device/wrs/alias/psa/splitpack/inline/shader.comp";

        std::map<std::string, std::string> defines;
        if (m_usePartitionElements) {
            defines["USE_PARTITION_ELEMENTS"];
        }

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {}, defines);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        specInfoBuilder.add_entry(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        specInfoBuilder.add_entry(m_splitSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("Inline-SplitPack");
            profiler.value()->cmd_start(cmd, "Inline-SplitPack");
        }
#endif

        host::glsl::uint K = N / m_splitSize;

        cmd->bind(m_pipeline);
        if (m_usePartitionElements) {
            cmd->push_descriptor_set(m_pipeline,
                                     buffers.weights,          //
                                     buffers.partitionIndices, //
                                     buffers.partitionPrefix,  //
                                     buffers.heavyCount,       //
                                     buffers.mean,             //
                                     buffers.aliasTable,       //
                                     buffers.partitionElements);
        } else {
            cmd->push_descriptor_set(m_pipeline,               //
                                     buffers.weights,          //
                                     buffers.partitionIndices, //
                                     buffers.partitionPrefix,  //
                                     buffers.heavyCount,       //
                                     buffers.mean,             //
                                     buffers.aliasTable);
        }
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .K = K,
                                                          .N = N,
                                                      });
        const host::glsl::uint splitsPerDispatch = m_workgroupSize - 1;
        const host::glsl::uint workgroupCount = (K + splitsPerDispatch - 1) / splitsPerDispatch;
        cmd->dispatch(workgroupCount, 1, 1);

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif
    }

  private:
    merian::PipelineHandle m_pipeline;
    const host::glsl::uint m_workgroupSize;
    const host::glsl::uint m_splitSize;
    const bool m_usePartitionElements;
};

} // namespace device
