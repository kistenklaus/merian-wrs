#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : svo.hpp
 *
 * Small value optimization (SVO) for the hierarchical sampling approach,
 * replaces the iterations of the HSTC and HSTSampling kernels, which require only
 * a single workgroup to be performed within a single workgroup.
 */

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/algorithm/hs/HSTRepr.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct HSSVOBuffers {
    using Self = HSSVOBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle hst;
    using HstLayout = layout::ArrayLayout<float, storageQualifier>;
    using HstView = layout::BufferView<HstLayout>;

    merian::BufferHandle histogram;
    using HistogramLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using HistogramView = layout::BufferView<HistogramLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N) {
        wrs::hst::HSTRepr repr{N};
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.hst = alloc->createBuffer(HstLayout::size(repr.size()),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
            buffers.histogram = alloc->createBuffer(HistogramLayout::size(repr.size()),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferSrc,
                                                    memoryMapping);
        } else {
            buffers.hst = alloc->createBuffer(HstLayout::size(repr.size()),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferSrc,
                                              memoryMapping);
            buffers.histogram = alloc->createBuffer(HistogramLayout::size(repr.size()),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferDst,
                                                    memoryMapping);
        }
        return buffers;
    }
};

// Hierarchical Sampling Small Value Optimization
class HSSVO {
    struct PushConstants {
        glsl::uint S;
        glsl::uint regionOffset;
        glsl::uint regionSize;
    };

  public:
    using Buffers = HSSVOBuffers;

    explicit HSSVO(const merian::ContextHandle& context,
                   const merian::ShaderCompilerHandle& shaderCompiler,
                   glsl::uint workgroupSize)
        : m_workgroupSize(workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/hs/svo/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             glsl::uint S,
             glsl::uint regionOffset,
             glsl::uint regionSize) const {

        /* fmt::println("SVO: offset = {}, size = {}", regionOffset, regionSize); */

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.hst, buffers.histogram);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .S = S,
                                                          .regionOffset = regionOffset,
                                                          .regionSize = regionSize,
                                                      });
        const uint32_t workgroupCount = 1;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    glsl::uint getWorkgroupSize() const {
        return m_workgroupSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
