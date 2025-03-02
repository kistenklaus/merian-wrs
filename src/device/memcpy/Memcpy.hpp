#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device {

template <typename T = float> struct MemcpyBuffers {
    using Self = MemcpyBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle src;
    using SrcLayout = host::layout::ArrayLayout<T, storageQualifier>;
    using SrcView = host::layout::BufferView<SrcLayout>;

    merian::BufferHandle dst;
    using DstLayout = host::layout::ArrayLayout<T, storageQualifier>;
    using DstView = host::layout::BufferView<DstLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.src = alloc->createBuffer(SrcLayout::size(N),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst |
                                                  vk::BufferUsageFlagBits::eTransferSrc,
                                              memoryMapping, "src");
            buffers.dst = alloc->createBuffer(DstLayout::size(N),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferSrc |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping, "dst");

        } else {
            buffers.src = alloc->createBuffer(SrcLayout::size(N),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferSrc,
                                              memoryMapping, "src-stage");
            buffers.dst = alloc->createBuffer(DstLayout::size(N),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping, "dst-stage");
        }
        return buffers;
    }
};

struct MemcpyConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;

    explicit constexpr MemcpyConfig(host::glsl::uint workgroupSize, host::glsl::uint rows)
        : workgroupSize(workgroupSize), rows(rows) {}

    constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

template <typename T = float> class Memcpy {
    struct PushConstants {
        host::glsl::uint N;
    };

  public:
    using Buffers = MemcpyBuffers<T>;

    explicit Memcpy(const merian::ContextHandle& context,
                    const merian::ShaderCompilerHandle& shaderCompiler,
                    MemcpyConfig config)
        : m_blockSize(config.blockSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // reductions
                .add_binding_storage_buffer() // prefix sum
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/memcpy/shader.comp";

        std::map<std::string, std::string> defines;

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {"src/device/common/"},
            defines);

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

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.src, buffers.dst);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;

        cmd->dispatch(workgroupCount, 1, 1);

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
          profiler.value()->start("Memcpy");
          profiler.value()->cmd_start(cmd, "Memcpy", vk::PipelineStageFlagBits::eTopOfPipe);
        }
#endif


#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
          profiler.value()->end();
          profiler.value()->cmd_end(cmd, vk::PipelineStageFlagBits::eComputeShader);
        }
#endif
    }

    inline host::glsl::uint blockSize() const {
        return m_blockSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_blockSize;
};

} // namespace device
