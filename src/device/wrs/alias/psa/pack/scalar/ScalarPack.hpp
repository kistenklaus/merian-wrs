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
#include <fmt/base.h>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct ScalarPackBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using weight_type = host::glsl::f32;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<host::glsl::uint, "heavyCount">,
                                   host::layout::Attribute<host::glsl::uint, "__padding1">,
                                   host::layout::Attribute<host::glsl::uint, "__padding2">,
                                   host::layout::Attribute<host::glsl::uint, "__padding3">,
                                   host::layout::Attribute<host::glsl::uint*, "heavyLightIndices">>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle heavyCount;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean;
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle splits;
    using SplitStructLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<host::glsl::uint, "i">,
                                   host::layout::Attribute<host::glsl::uint, "j">,
                                   host::layout::Attribute<weight_type, "spill">>;
    using SplitsLayout = host::layout::ArrayLayout<SplitStructLayout, storageQualifier>;
    using SplitsView = host::layout::BufferView<SplitsLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableEntryLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<weight_type, "p">,
                                   host::layout::Attribute<host::glsl::uint, "a">>;
    using AliasTableLayout = host::layout::ArrayLayout<AliasTableEntryLayout, storageQualifier>;
    using AliasTableView = host::layout::BufferView<AliasTableLayout>;

    merian::BufferHandle partitionElements;

    static ScalarPackBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                      std::size_t weightCount,
                                      std::size_t splitCount,
                                      merian::MemoryMappingType memoryMapping);
};

class ScalarPackConfig {
  public:
    const host::glsl::uint workgroupSize;
    const host::glsl::uint splitSize;

    constexpr ScalarPackConfig() : workgroupSize(512), splitSize(2) {}
    explicit constexpr ScalarPackConfig(host::glsl::uint splitSize,
                                        host::glsl::uint workgroupSize = 512)
        : workgroupSize(workgroupSize), splitSize(splitSize) {}
};

class ScalarPack {

    struct PushConstant {
        host::glsl::uint N;
        host::glsl::uint K;
    };

  public:
    using Buffers = ScalarPackBuffers;
    using Config = ScalarPackConfig;
    using weight_t = float;

    explicit ScalarPack(const merian::ContextHandle& context,
                        const merian::ShaderCompilerHandle& shaderCompiler,
                        ScalarPackConfig config,
                        bool usePartitionElements)
        : m_workgroupSize(config.workgroupSize), m_splitSize(config.splitSize),
          m_usePartitionElements(usePartitionElements) {

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

        std::string shaderPath = "src/device/wrs/alias/psa/pack/scalar/float.comp";

        std::map<std::string, std::string> defines;
        if (m_usePartitionElements) {
            defines["USE_PARTITION_ELEMENTS"];
        }

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {}, defines);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstant>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry<host::glsl::uint>(config.workgroupSize);
        specInfoBuilder.add_entry<host::glsl::uint>(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const ScalarPackBuffers& buffers,
             const host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("ScalarPack");
            profiler.value()->cmd_start(cmd, "ScalarPack");
        }
#endif

        const host::glsl::uint K = (N + m_splitSize - 1) / m_splitSize;

        cmd->bind(m_pipeline);
        if (m_usePartitionElements) {
            cmd->push_descriptor_set(m_pipeline,
                                     buffers.partitionIndices, // binding = 0
                                     buffers.heavyCount,       // binding = 1
                                     buffers.weights,          // binding = 2
                                     buffers.mean,             // binding = 3
                                     buffers.splits,           // binding = 4
                                     buffers.aliasTable,       // binding = 5
                                     buffers.partitionElements // binding = 6

            );
        } else {
            cmd->push_descriptor_set(m_pipeline,
                                     buffers.partitionIndices, // binding = 0
                                     buffers.heavyCount,       // binding = 1
                                     buffers.weights,          // binding = 2
                                     buffers.mean,             // binding = 3
                                     buffers.splits,           // binding = 4
                                     buffers.aliasTable        // binding = 5
            );
        }
        cmd->push_constant<PushConstant>(m_pipeline, PushConstant{.N = N, .K = K});
        const uint32_t workgroupCount = (K + m_workgroupSize - 1) / m_workgroupSize;
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
    host::glsl::uint m_workgroupSize;
    host::glsl::uint m_splitSize;
    const bool m_usePartitionElements;
};

} // namespace device
