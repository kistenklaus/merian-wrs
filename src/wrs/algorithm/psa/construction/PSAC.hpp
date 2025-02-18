#pragma once
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/algorithm/mean/atomic/AtomicMean.hpp"
#include "src/wrs/algorithm/splitpack/SplitPack.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <merian/vk/utils/profiler.hpp>
#include <src/wrs/algorithm/pack/scalar/ScalarPack.hpp>
#include <src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartition.hpp>
#include <src/wrs/algorithm/split/scalar/ScalarSplit.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {
struct PSACBuffers {
    using weight_type = glsl::f32;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean;
    using MeanLayout = layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = layout::BufferView<MeanLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = DecoupledPrefixPartitionBuffers::PartitionLayout;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout = DecoupledPrefixPartitionBuffers::PartitionPrefixLayout;
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle partitionDecoupledState;
    using PartitionDecoupledStateLayout = DecoupledPrefixPartitionBuffers::BatchDescriptorsLayout;
    using PartitionDecoupledStateView = layout::BufferView<PartitionDecoupledStateLayout>;

    merian::BufferHandle splits;
    using SplitsLayout = ScalarSplitBuffers::SplitsLayout;
    using SplitsView = layout::BufferView<SplitsLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableLayout = ScalarPackBuffers::AliasTableLayout;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    static PSACBuffers allocate(const merian::ResourceAllocatorHandle& alloc,
                                std::size_t weightCount,
                                std::size_t meanPartitionSize,
                                std::size_t prefixPartitionSize,
                                std::size_t splitCount,
                                merian::MemoryMappingType memoryMapping);
};

struct PSACConfig {
    AtomicMeanConfig meanConfig;
    DecoupledPrefixPartitionConfig prefixPartitionConfig;
    glsl::uint splitWorkgroupSize;
    glsl::uint packWorkgroupSize;

    glsl::uint splitSize;

    constexpr PSACConfig()
        : meanConfig{}, prefixPartitionConfig{}, splitWorkgroupSize(512), packWorkgroupSize(512),
          splitSize(2) {}
};

class PSAC {

  public:
    using Buffers = PSACBuffers;
    using weight_t = glsl::f32;

    explicit PSAC(const merian::ContextHandle& context,
                  const merian::ShaderCompilerHandle& shaderCompiler,
                  PSACConfig config = {})
        : m_mean(context, shaderCompiler, config.meanConfig),
          /* m_mean(context, config.meanWorkgroupSize, config.meanRows, false), */
          m_prefixPartition(context, shaderCompiler, config.prefixPartitionConfig),
          m_splitpack(context, shaderCompiler, config.splitWorkgroupSize, config.splitSize),
          /* m_split(context, config.splitWorkgroupSize), m_pack(context, config.packWorkgroupSize),
           */
          m_splitSize(config.splitSize) {}

    void run(const merian::CommandBufferHandle cmd,
             const Buffers& buffers,
             const std::size_t weightCount,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {

        AtomicMeanBuffers meanBuffers;
        meanBuffers.elements = buffers.weights;
        meanBuffers.mean = buffers.mean;

        DecoupledPrefixPartitionBuffers partitionBuffers;
        partitionBuffers.elements = buffers.weights;
        partitionBuffers.pivot = buffers.mean;
        partitionBuffers.partition = buffers.partitionIndices;
        partitionBuffers.partitionPrefix = buffers.partitionPrefix;
        partitionBuffers.batchDescriptors = buffers.partitionDecoupledState;

        std::size_t prefixPartitionSize = m_prefixPartition.getPartitionSize();
        std::size_t prefixWorkgroupCount =
            (weightCount + prefixPartitionSize - 1) / prefixPartitionSize;
        DecoupledPrefixPartitionBuffers::BatchDescriptorsView prefixStates{
            partitionBuffers.batchDescriptors, prefixWorkgroupCount};

        AtomicMeanBuffers::MeanView meanView{buffers.mean};

        if (profiler.has_value()) {
            profiler.value()->start("Prepare");
            profiler.value()->cmd_start(cmd, "Prepare");
        }

        prefixStates.zero(cmd);
        meanView.zero(cmd);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        prefixStates.expectComputeRead(cmd);
        meanView.expectComputeRead(cmd);

        if (profiler.has_value()) {
            MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Mean");
            m_mean.run(cmd, meanBuffers, weightCount);
        } else {
            m_mean.run(cmd, meanBuffers, weightCount);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.mean->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eShaderRead));

        if (profiler.has_value()) {
            MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "PrefixPartition");
            m_prefixPartition.run(cmd, partitionBuffers, weightCount);
        } else {
            m_prefixPartition.run(cmd, partitionBuffers, weightCount);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {
                         buffers.partitionIndices->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                  vk::AccessFlagBits::eShaderRead),
                         buffers.partitionPrefix->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead),
                     });

        SplitPackBuffers splitPackBuffers;
        splitPackBuffers.weights = buffers.weights;
        splitPackBuffers.partitionIndices = buffers.partitionIndices;
        splitPackBuffers.partitionPrefix = buffers.partitionPrefix;
        splitPackBuffers.mean = buffers.mean;
        splitPackBuffers.aliasTable = buffers.aliasTable;
        splitPackBuffers.splits = buffers.splits;

        if (profiler.has_value()) {
            profiler.value()->start("SplitPack");
            profiler.value()->cmd_start(cmd, "SplitPack");
        }
        m_splitpack.run(cmd, splitPackBuffers, weightCount);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

    inline glsl::uint getPrefixPartitionSize() const {
        return m_prefixPartition.getPartitionSize();
    }

    /*inline glsl::uint getMeanPartitionSize() const {*/
    /*    return m_mean.getPartitionSize();*/
    /*}*/

    inline glsl::uint getSplitSize() const {
        return m_splitSize;
    }

  private:
    AtomicMean m_mean;
    DecoupledPrefixPartition m_prefixPartition;
    /*ScalarSplit m_split;*/
    /*ScalarPack m_pack;*/
    SplitPack m_splitpack;

    glsl::uint m_splitSize;
};
} // namespace wrs
