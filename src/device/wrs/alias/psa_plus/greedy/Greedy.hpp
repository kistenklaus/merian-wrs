#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/device/wrs/alias/psa_plus/layout/heavy_light_count.hpp"
#include "src/host/layout/Attribute.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/layout/StructLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/base.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device {

struct PSAGreedyBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using weight_type = host::glsl::f32;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean;
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle heavyLightCount;
    using HeavyLightCountView = host::layout::BufferView<device::details::HeavyLightCountLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableEntryLayout =
        host::layout::StructLayout<storageQualifier,
                                   host::layout::Attribute<weight_type, "p">,
                                   host::layout::Attribute<host::glsl::uint, "a">>;
    using AliasTableLayout = device::details::AliasTableLayout;
    using AliasTableView = host::layout::BufferView<AliasTableLayout>;

    merian::BufferHandle decoupledStates;

    merian::BufferHandle debug;
    using DebugLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using DebugView = host::layout::BufferView<DebugLayout>;

    static PSAGreedyBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                     std::size_t N,
                                     std::size_t blockSize,
                                     merian::MemoryMappingType memoryMapping) {
        PSAGreedyBuffers buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferDst |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);
            buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               memoryMapping);
            buffers.heavyLightCount = device::details::allocateHeavyLightBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);

            buffers.partitionIndices = alloc->createBuffer(
                PartitionIndicesLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                memoryMapping);
            buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout::size(N),
                                                          vk::BufferUsageFlagBits::eStorageBuffer |
                                                              vk::BufferUsageFlagBits::eTransferSrc,
                                                          memoryMapping);

            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, N);

            std::size_t blocks = (blockSize + N - 1) / blockSize;

            buffers.decoupledStates = alloc->createBuffer(
                blocks * 40 + 4, vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);

            buffers.debug = alloc->createBuffer(DebugLayout::size(N),
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferSrc,
                                                memoryMapping);

        } else {
            buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                                  vk::BufferUsageFlagBits::eTransferSrc |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                  memoryMapping);
            buffers.mean = alloc->createBuffer(
                MeanLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);

            buffers.heavyLightCount = device::details::allocateHeavyLightBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eTransferDst);

            buffers.partitionIndices =
                alloc->createBuffer(PartitionIndicesLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.partitionPrefix =
                alloc->createBuffer(PartitionPrefixLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping, vk::BufferUsageFlagBits::eTransferDst, N);

            buffers.debug = alloc->createBuffer(
                DebugLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }

        return buffers;
    }
};

class PSAGreedyConfig {
  public:
    host::glsl::uint workgroupSize;
    host::glsl::uint rows;

    explicit constexpr PSAGreedyConfig(host::glsl::uint rows, host::glsl::uint workgroupSize = 512)
        : workgroupSize(workgroupSize), rows(rows) {}

    host::glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

class PSAGreedy {

    struct PushConstant {
        host::glsl::uint N;
    };

  public:
    using Buffers = PSAGreedyBuffers;
    using Config = PSAGreedyConfig;
    using weight_t = float;

    explicit PSAGreedy(const merian::ContextHandle& context,
                       const merian::ShaderCompilerHandle& shaderCompiler,
                       Config config)
        : m_blockSize(config.blockSize()) {

        auto setBuilder = merian::DescriptorSetLayoutBuilder()
                              .add_binding_storage_buffer()  // weights
                              .add_binding_storage_buffer()  // mean
                              .add_binding_storage_buffer()  // heavy & light count
                              .add_binding_storage_buffer()  // partition indices
                              .add_binding_storage_buffer()  // partition prefix
                              .add_binding_storage_buffer()  // alias table
                              .add_binding_storage_buffer()  // decoupled states
                              .add_binding_storage_buffer(); // debug

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            setBuilder.build_push_descriptor_layout(context);

        std::string shaderPath = "src/device/wrs/alias/psa_plus/greedy/shader.comp";

        std::map<std::string, std::string> defines;
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
        specInfoBuilder.add_entry<host::glsl::uint>(config.rows);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             const host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {

        cmd->fill(buffers.decoupledStates);

        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.decoupledStates->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                             vk::AccessFlagBits::eShaderRead));

        cmd->bind(m_pipeline);

        cmd->push_descriptor_set(m_pipeline, buffers.weights, buffers.mean, buffers.heavyLightCount,
                                 buffers.partitionIndices, buffers.partitionPrefix,
                                 buffers.aliasTable, buffers.decoupledStates, buffers.debug);

        cmd->push_constant<PushConstant>(m_pipeline, PushConstant{
                                                         .N = N,
                                                     });

        host::glsl::uint workgroupCount = (N + m_blockSize - 1) / m_blockSize;
        cmd->dispatch(workgroupCount);
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_blockSize;
};

} // namespace device
