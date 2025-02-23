#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/base.h>
#include <memory>
#include <tuple>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>

namespace device {

struct ScalarSplitBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using weight_type = host::glsl::f32;

    merian::BufferHandle partitionPrefix;

    merian::BufferHandle heavyCount;

    merian::BufferHandle mean;

    merian::BufferHandle splits;
};

struct ScalarSplitConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint splitSize;

    constexpr explicit ScalarSplitConfig(host::glsl::uint splitSize,
                                         host::glsl::uint workgroupSize = 512)
        : workgroupSize(workgroupSize), splitSize(splitSize) {}
};

class ScalarSplit {

    struct PushConstants {
        host::glsl::uint K;
        host::glsl::uint N;
    };

  public:
    using Buffers = ScalarSplitBuffers;
    using Config = ScalarSplitConfig;
    using weight_t = float;

    ScalarSplit(const merian::ContextHandle& context,
                const merian::ShaderCompilerHandle& shaderCompiler,
                Config config)
        : m_workgroupSize(config.workgroupSize), m_splitSize(config.splitSize) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // partition prefix sums
                .add_binding_storage_buffer() // partition heavy
                .add_binding_storage_buffer() // mean
                .add_binding_storage_buffer() // splits
                .build_push_descriptor_layout(context);

        std::string shaderPath = "src/device/wrs/alias/psa/split/scalar/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<std::tuple<uint32_t, uint32_t>>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const ScalarSplitBuffers& buffers,
             uint32_t N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("ScalarSplit");
            profiler.value()->cmd_start(cmd, "ScalarSplit");
        }
#endif

        const host::glsl::uint K = (N + m_splitSize - 1) / m_splitSize;
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline,
                                 buffers.partitionPrefix, // binding = 0
                                 buffers.heavyCount,      // binding = 1
                                 buffers.mean,            // binding = 2
                                 buffers.splits           // binding = 3
        );
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .K = K,
                                                          .N = N,
                                                      });
        const host::glsl::uint workgroupCount = (K - 1 + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif
    }

    inline host::glsl::uint splitSize() const {
        return m_splitSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    std::vector<vk::WriteDescriptorSet> m_writes;
    host::glsl::uint m_workgroupSize;
    host::glsl::uint m_splitSize;
};

} // namespace device
