#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/host/layout/Attribute.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/layout/StructLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <bit>
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct SubgroupPackBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using weight_type = host::glsl::f32;

    merian::BufferHandle partitionIndices; // bind = 0
    using PartitionIndicesLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<host::glsl::uint, "heavyCount">,
                                   host::layout::Attribute<host::glsl::uint*, "heavyLightIndices">>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle heavyCount;

    merian::BufferHandle weights; // binding = 1
    using WeightsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean; // binding = 2
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle splits; // binding = 3
    using SplitStructLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<host::glsl::uint, "i">,
                                   host::layout::Attribute<host::glsl::uint, "j">,
                                   host::layout::Attribute<weight_type, "spill">>;
    using SplitsLayout = host::layout::ArrayLayout<SplitStructLayout, storageQualifier>;
    using SplitsView = host::layout::BufferView<SplitsLayout>;

    merian::BufferHandle aliasTable; // binding = 4
    using AliasTableEntryLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<weight_type, "p">,
                                   host::layout::Attribute<host::glsl::uint, "a">>;
    using AliasTableLayout = host::layout::ArrayLayout<AliasTableEntryLayout, storageQualifier>;
    using AliasTableView = host::layout::BufferView<AliasTableLayout>;

    merian::BufferHandle partition; // binding = 5
    using PartitionLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using PartitionView = host::layout::BufferView<PartitionLayout>;

    merian::BufferHandle partitionPrefix; // binding = 6
    using PartitionPrefixLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout>;

    static SubgroupPackBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                        std::size_t weightCount,
                                        std::size_t splitCount,
                                        merian::MemoryMappingType memoryMapping);
};

struct SubgroupPackConfig {
    const host::glsl::uint splitSize; // K
    const host::glsl::uint workgroupSize;
    const host::glsl::uint subgroupSplit;

    constexpr SubgroupPackConfig() : splitSize(2), workgroupSize(512), subgroupSplit(4) {}
    explicit constexpr SubgroupPackConfig(host::glsl::uint splitSize,
                                          host::glsl::uint subgroupSplit = 4,
                                          host::glsl::uint workgroupSize = 512)
        : splitSize(splitSize), workgroupSize(workgroupSize), subgroupSplit(subgroupSplit) {}
};

class SubgroupPack {
    struct PushConstants {
        host::glsl::uint N;
        host::glsl::uint K;
    };

  public:
    using Buffers = SubgroupPackBuffers;
    using Config = SubgroupPackConfig;

    explicit SubgroupPack(const merian::ContextHandle& context,
                          const merian::ShaderCompilerHandle& shaderCompiler,
                          SubgroupPackConfig config,
                          bool usePartitionElements)
        : m_splitSize(config.splitSize), m_usePartitionElements(usePartitionElements) {

        auto setBuilder = merian::DescriptorSetLayoutBuilder()
                              .add_binding_storage_buffer()  // partition indices
                              .add_binding_storage_buffer()  // heavy count
                              .add_binding_storage_buffer()  // weights
                              .add_binding_storage_buffer()  // mean
                              .add_binding_storage_buffer()  // splits
                              .add_binding_storage_buffer(); // alias table
        if (m_usePartitionElements) {
            setBuilder.add_binding_storage_buffer(); // partition elements
        }

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            setBuilder.build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/wrs/alias/psa/pack/subgroup/shader.comp";

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
        specInfoBuilder.add_entry(config.workgroupSize); // 0
        host::glsl::uint subgroupSize =
            context->physical_device.physical_device_subgroup_properties.subgroupSize;
        assert(config.workgroupSize % subgroupSize == 0);
        specInfoBuilder.add_entry(subgroupSize); // 1
        /* glsl::uint log2SubgroupSize = std::bit_width(subgroupSize) - 1; // floor(log2( . )) */
        /* specInfoBuilder.add_entry(log2SubgroupSize); */

        host::glsl::uint threadsPerSubproblem = subgroupSize / config.subgroupSplit;
        specInfoBuilder.add_entry(threadsPerSubproblem); // 2
        host::glsl::uint log2ThreadsPerSubgroup = std::bit_width(threadsPerSubproblem) - 1;
        specInfoBuilder.add_entry(log2ThreadsPerSubgroup); // 3

        specInfoBuilder.add_entry(config.splitSize); // 4

        host::glsl::uint subgroupCount = (config.workgroupSize + subgroupSize - 1) / subgroupSize;
        m_subproblemsPerWorkgroup = subgroupCount * config.subgroupSplit;

        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("SubgroupPack");
            profiler.value()->cmd_start(cmd, "SubgroupPack");
        }
#endif

        cmd->bind(m_pipeline);
        if (m_usePartitionElements) {
            assert(buffers.partition != nullptr);
            cmd->push_descriptor_set(m_pipeline, buffers.partitionIndices, buffers.heavyCount,
                                     buffers.weights, buffers.mean, buffers.splits,
                                     buffers.aliasTable, buffers.partition);
        } else {
            cmd->push_descriptor_set(m_pipeline, buffers.partitionIndices, buffers.heavyCount,
                                     buffers.weights, buffers.mean, buffers.splits,
                                     buffers.aliasTable);
        }

        host::glsl::uint K = ((N + m_splitSize - 1) / m_splitSize);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .N = N,
                                                          .K = K,
                                                      });
        const uint32_t workgroupCount =
            (K + m_subproblemsPerWorkgroup - 1) / m_subproblemsPerWorkgroup;
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
    host::glsl::uint m_subproblemsPerWorkgroup;
    host::glsl::uint m_splitSize;
    bool m_usePartitionElements;
};

} // namespace device
