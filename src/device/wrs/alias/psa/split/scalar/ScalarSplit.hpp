#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/alias/psa/layout/split.hpp"
#include "src/device/wrs/alias/psa/split/SplitAllocFlags.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <fmt/base.h>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>

namespace device {

struct ScalarSplitBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using weight_type = host::glsl::f32;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    merian::BufferHandle mean;
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle splits;
    using SplitsLayout = device::details::SplitsLayout;
    using SplitsView = host::layout::BufferView<SplitsLayout>;

    static ScalarSplitBuffers allocate(const merian::ResourceAllocatorHandle& alloc,
                                       merian::MemoryMappingType memoryMapping,
                                       std::size_t N,
                                       std::size_t splitSize,
                                       SplitAllocFlags allocFlags = SplitAllocFlags::ALLOC_ALL) {
        ScalarSplitBuffers buffers;
        std::size_t K = (N + splitSize - 1) / splitSize;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            if ((allocFlags & SplitAllocFlags::ALLOC_PARTITION_PREFIX) != 0) {
                buffers.partitionPrefix =
                    alloc->createBuffer(PartitionPrefixLayout::size(N),
                                        vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
                buffers.heavyCount =
                    alloc->createBuffer(HeavyCountLayout::size(),
                                        vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_MEAN) != 0) {
                buffers.mean = alloc->createBuffer(
                    MeanLayout::size(), vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_SPLITS) != 0) {
                buffers.splits = device::details::allocateSplitBuffer(
                    alloc, merian::MemoryMappingType::NONE, vk::BufferUsageFlagBits::eStorageBuffer,
                    K);
            }
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
        return buffers;
    }
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
