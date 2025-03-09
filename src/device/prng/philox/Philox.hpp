#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct PhiloxBuffers {
    using Self = PhiloxBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle samples;
    using SamplesLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using SamplesView = host::layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint sampleCount) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.samples = alloc->createBuffer(SamplesLayout::size(sampleCount),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  merian::MemoryMappingType::NONE);
        } else {
            buffers.samples =
                alloc->createBuffer(SamplesLayout::size(sampleCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

class PhiloxConfig {
  public:
    host::glsl::uint workgroupSize;
    host::Distribution distribution;

    constexpr PhiloxConfig()
        : workgroupSize(128), distribution(host::Distribution::SEEDED_RANDOM_UNIFORM) {}
    explicit constexpr PhiloxConfig(host::Distribution distribution, host::glsl::uint workgroupSize = 128)
        : workgroupSize(workgroupSize), distribution(distribution) {}
};

class Philox {
    struct PushConstants {
        host::glsl::uint seed;
    };

  public:
    using Buffers = PhiloxBuffers;

    explicit Philox(const merian::ContextHandle& context,
                    const merian::ShaderCompilerHandle& shaderCompiler,
                    PhiloxConfig config = {})
        : m_workgroupSize(config.workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/prng/philox/shader.comp";

        std::map<std::string, std::string> defines;
        if ((config.distribution == host::Distribution::PSEUDO_RANDOM_UNIFORM) |
            (config.distribution == host::Distribution::SEEDED_RANDOM_UNIFORM) |
            (config.distribution == host::Distribution::RANDOM_UNIFORM)) {
            defines["UNIFORM_DISTRIBUTION"];
        } else if (config.distribution == host::Distribution::SEEDED_RANDOM_EXPONENTIAL) {
            defines["EXP_DISTRIBUTION"];
        } else {
            throw std::runtime_error("Unsupported distribution");
        }

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {}, defines);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(config.workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint sampleCount,
             host::glsl::uint seed = 12345u) const {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.samples);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .seed = seed,
                                                      });
        const uint32_t workgroupCount = (sampleCount + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_workgroupSize;
};

} // namespace device
