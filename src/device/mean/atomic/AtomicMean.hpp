#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : AtomicMean.hpp
 *
 * Single dispatch mean using floating point atomic means.
 * A workgroup computes it's local mean, and publishes using atomics.
 *
 * This is the best performing implementation we found, however it does require floating point
 * atomics.
 */

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/mean/MeanAllocFlags.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/layout/PrimitiveLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/format.h>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct AtomicMeanBuffers {
    using Self = AtomicMeanBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = host::layout::BufferView<ElementsLayout>;

    merian::BufferHandle mean;
    using MeanLayout = host::layout::PrimitiveLayout<float, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N,
                         MeanAllocFlags allocFlags = MeanAllocFlags::ALLOC_ALL) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            if ((allocFlags & MeanAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       memoryMapping);
            }
            if ((allocFlags & MeanAllocFlags::ALLOC_MEAN) != 0) {
                buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferSrc,
                                                   memoryMapping);
            }

        } else {
            if ((allocFlags & MeanAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements = alloc->createBuffer(
                    ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }
            if ((allocFlags & MeanAllocFlags::ALLOC_MEAN) != 0) {
                buffers.mean = alloc->createBuffer(
                    MeanLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }
        }
        return buffers;
    }
};

class AtomicMeanConfig {
  public:
    host::glsl::uint workgroupSize;
    host::glsl::uint rows;

    constexpr AtomicMeanConfig() : workgroupSize(512), rows(8) {}
    explicit constexpr AtomicMeanConfig(host::glsl::uint workgroupSize, host::glsl::uint rows)
        : workgroupSize(workgroupSize), rows(rows) {}

    inline constexpr host::glsl::uint partitionSize() const {
        return workgroupSize * rows;
    }
};

class AtomicMean {
    struct PushConstants {
        host::glsl::uint N;
    };

  public:
    using Buffers = AtomicMeanBuffers;

    explicit AtomicMean(const merian::ContextHandle& context,
                        const merian::ShaderCompilerHandle& shaderCompiler,
                        AtomicMeanConfig config = {})
        : m_partitionSize(config.rows * config.workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // mean
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/mean/atomic/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(config.workgroupSize);
        specInfoBuilder.add_entry(config.rows);
        specInfoBuilder.add_entry(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N) {

        cmd->fill(buffers.mean, 0);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.mean->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                  vk::AccessFlagBits::eShaderRead));

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.mean);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .N = N,
                                                      });
        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_partitionSize;
};

} // namespace device
