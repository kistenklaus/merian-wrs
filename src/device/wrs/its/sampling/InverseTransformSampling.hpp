#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : InverseTransformSampling.hpp
 *
 * The actual sampling step of the ITS method.
 * Performs a binary search over the CMF and writes the samples
 * into a global memory buffer.
 * The ITS method can be suprisingly fast for sample sizes smaller than the weight sizes.
 */

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
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
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct InverseTransformSamplingBuffers {
    using Self = InverseTransformSamplingBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle cmf; // cummulative mass function
    using CMFLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using CMFView = host::layout::BufferView<CMFLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using SamplesView = host::layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t cmfSize,
                         std::size_t sampleCount) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.cmf = alloc->createBuffer(CMFLayout::size(cmfSize),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              merian::MemoryMappingType::NONE);
            buffers.samples = alloc->createBuffer(SamplesLayout::size(sampleCount),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  merian::MemoryMappingType::NONE);
        } else {
            buffers.cmf = alloc->createBuffer(CMFLayout::size(cmfSize),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.samples =
                alloc->createBuffer(SamplesLayout::size(sampleCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

class InverseTransformSamplingConfig {
  public:
    host::glsl::uint workgroupSize;
    host::glsl::uint cooperativeSamplingSize;
    bool pArraySearch;

    constexpr InverseTransformSamplingConfig()
        : workgroupSize(512), cooperativeSamplingSize(4096), pArraySearch(true) {}
    explicit constexpr InverseTransformSamplingConfig(host::glsl::uint workgroupSize,
                                                      host::glsl::uint cooperativeSamplingSize,
                                                      bool pArraySearch = false)
        : workgroupSize(workgroupSize), cooperativeSamplingSize(cooperativeSamplingSize), pArraySearch(pArraySearch) {}
};

class InverseTransformSampling {
    struct PushConstants {
        host::glsl::uint N; // cmf size
        host::glsl::uint S; // sample count
        host::glsl::uint seed;
    };

  public:
    using Buffers = InverseTransformSamplingBuffers;

    explicit InverseTransformSampling(const merian::ContextHandle& context,
                                      const merian::ShaderCompilerHandle& shaderCompiler,
                                      InverseTransformSamplingConfig config = {})
        : m_workgroupSize(config.workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/wrs/its/sampling/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        specInfoBuilder.add_entry(config.cooperativeSamplingSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             host::glsl::uint S,
             host::glsl::uint seed = 12345u) const {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.cmf, buffers.samples);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .N = N,
                                                          .S = S,
                                                          .seed = seed,
                                                      });
        const uint32_t workgroupCount = (S + m_workgroupSize - 1) / m_workgroupSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_workgroupSize;
};

} // namespace device
