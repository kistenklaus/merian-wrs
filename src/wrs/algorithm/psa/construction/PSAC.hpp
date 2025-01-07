#pragma once
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <merian/vk/utils/profiler.hpp>
#include <src/wrs/algorithm/pack/scalar/ScalarPack.hpp>
#include <src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartition.hpp>
#include <src/wrs/algorithm/split/scalar/ScalarSplit.hpp>
#include <src/wrs/common_vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
#include "src/wrs/algorithm/mean/decoupled/DecoupledMean.hpp"

namespace wrs {
struct PSACBuffers {

    static constexpr uint32_t DEFAULT_WORKGROUP_SIZE = 512;
    static constexpr uint32_t DEFAULT_ROWS = 4;
    using weight_type = glsl::float_t;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle meanDecoupledStates;
    using MeanDecoupledStatesLayout = DecoupledMeanBuffers::DecoupledStatesLayout;
    using MeanDecoupledStateView = layout::BufferView<MeanDecoupledStatesLayout>;

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
    using SplitLayout = ScalarSplitBuffers::SplitsLayout;
    using SplitView = layout::BufferView<SplitLayout>;

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
    glsl::uint meanWorkgroupSize;
    glsl::uint meanRows;
    glsl::uint prefixSumWorkgroupSize;
    glsl::uint prefixSumRows;
    glsl::uint splitWorkgroupSize;
    glsl::uint packWorkgroupSize;

    constexpr PSACConfig()
        : meanWorkgroupSize(512), meanRows(4), prefixSumWorkgroupSize(512), prefixSumRows(4),
          splitWorkgroupSize(512), packWorkgroupSize(512) {}
};

class PSAC {

  public:
    using Buffers = PSACBuffers;
    using weight_t = glsl::float_t;
    static constexpr uint32_t DEFAULT_WORKGROUP_SIZE = Buffers::DEFAULT_WORKGROUP_SIZE;
    static constexpr uint32_t DEFAULT_ROWS = Buffers::DEFAULT_ROWS;

    explicit PSAC(const merian::ContextHandle& context, PSACConfig config = {})
        : m_mean(context, config.meanWorkgroupSize, config.meanRows, false),
          m_prefixPartition(context, config.prefixSumWorkgroupSize, config.prefixSumRows, true, true),
          m_split(context, config.splitWorkgroupSize), m_pack(context, config.packWorkgroupSize) {}

    void run(const vk::CommandBuffer cmd,
             const Buffers& buffers,
             const std::size_t weightCount,
             const std::size_t splitCount,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {

        DecoupledMeanBuffers meanBuffers;
        meanBuffers.elements = buffers.weights;
        meanBuffers.decoupledStates = buffers.meanDecoupledStates;
        meanBuffers.mean = buffers.mean;

        DecoupledPrefixPartitionBuffers partitionBuffers;
        partitionBuffers.elements = buffers.weights;
        partitionBuffers.pivot = buffers.mean;
        partitionBuffers.partition = buffers.partitionIndices;
        partitionBuffers.partitionPrefix = buffers.partitionPrefix;
        partitionBuffers.batchDescriptors = buffers.partitionDecoupledState;

        std::size_t meanPartitionSize = m_mean.getPartitionSize();
        std::size_t meanWorkgroupCount = (weightCount + meanPartitionSize - 1) / meanPartitionSize;
        DecoupledMeanBuffers::DecoupledStatesView meanStates{meanBuffers.decoupledStates, meanWorkgroupCount};

        std::size_t prefixPartitionSize = m_prefixPartition.getPartitionSize();
        std::size_t prefixWorkgroupCount = (weightCount + prefixPartitionSize - 1) / prefixPartitionSize;
        DecoupledPrefixPartitionBuffers::BatchDescriptorsView prefixStates{partitionBuffers.batchDescriptors, prefixWorkgroupCount};


        if (profiler.has_value()) {
          profiler.value()->start("Prepare");
          profiler.value()->cmd_start(cmd, "Prepare");
        }

        meanStates.zero(cmd);
        prefixStates.zero(cmd);

        if (profiler.has_value()) {
          profiler.value()->end();
          profiler.value()->cmd_end(cmd);
        }
      
        meanStates.expectComputeRead(cmd);
        prefixStates.expectComputeRead(cmd);


        if (profiler.has_value()) {
            MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Mean");
            m_mean.run(cmd, meanBuffers, weightCount);
        } else {
            m_mean.run(cmd, meanBuffers, weightCount);
        }

        common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd, buffers.mean);


        if (profiler.has_value()) {
            MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "PrefixPartition");
            m_prefixPartition.run(cmd, partitionBuffers, weightCount);
        } else {
            m_prefixPartition.run(cmd, partitionBuffers, weightCount);
        }

        common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd, buffers.partitionIndices);
        common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd, buffers.partitionPrefix);

        ScalarSplitBuffers splitBuffers;
        splitBuffers.mean = buffers.mean;
        splitBuffers.splits = buffers.splits;
        splitBuffers.partitionPrefix = buffers.partitionPrefix;

        if (profiler.has_value()) {
            MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Split");
            m_split.run(cmd, splitBuffers, weightCount, splitCount);
        } else {
            m_split.run(cmd, splitBuffers, weightCount, splitCount);
        }

        common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd, buffers.splits);

        ScalarPackBuffers packBuffers;
        packBuffers.weights = buffers.weights;
        packBuffers.mean = buffers.mean;
        packBuffers.splits = buffers.splits;
        packBuffers.partitionIndices = buffers.partitionIndices;
        packBuffers.aliasTable = buffers.aliasTable;

        if (profiler.has_value()) {
            MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Pack");
            m_pack.run(cmd, weightCount, splitCount, packBuffers);
        } else {
            m_pack.run(cmd, weightCount, splitCount, packBuffers);
        }
    }

  private:
    DecoupledMean m_mean;
    DecoupledPrefixPartition m_prefixPartition;
    ScalarSplit m_split;
    ScalarPack m_pack;
};
} // namespace wrs
