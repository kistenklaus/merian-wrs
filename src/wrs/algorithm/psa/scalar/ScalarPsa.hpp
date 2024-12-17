#pragma once
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <memory>
#include <merian/vk/utils/profiler.hpp>
#include <src/wrs/common_vulkan.hpp>
#include <src/wrs/algorithm/pack/scalar/ScalarPack.hpp>
#include <src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartitionKernel.hpp>
#include <src/wrs/algorithm/split/scalar/ScalarSplit.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
#include "src/wrs/algorithm/mean/decoupled/DecoupledMean.h"

namespace wrs {
    struct ScalarPsaBuffers {

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
        using PartitionDecoupledStateLayout = DecoupledPrefixPartitionBuffers::BatchDescriptorsLayout;;
        using PartitionDecoupledStateView = layout::BufferView<PartitionDecoupledStateLayout>;

        merian::BufferHandle splits;
        using SplitLayout = ScalarSplitBuffers::SplitsLayout;
        using SplitView = layout::BufferView<SplitLayout>;

        merian::BufferHandle aliasTable;
        using AliasTableLayout = ScalarPackBuffers::AliasTableLayout;
        using AliasTableView = layout::BufferView<AliasTableLayout>;

        static ScalarPsaBuffers allocate(const merian::ResourceAllocatorHandle &alloc,
                                  const std::size_t weightCount,
                                  const std::size_t splitCount,
                                  const merian::MemoryMappingType memoryMapping
        ) {
            ScalarPsaBuffers buffers;
            std::size_t partitionSize = DEFAULT_WORKGROUP_SIZE * DEFAULT_ROWS;
            std::size_t workgroupCount = (weightCount + partitionSize - 1) / partitionSize;
            if (memoryMapping == merian::MemoryMappingType::NONE) {
                buffers.weights = alloc->createBuffer(WeightsLayout::size(weightCount),
                                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                      memoryMapping);
                buffers.meanDecoupledStates = alloc->createBuffer(MeanDecoupledStatesLayout::size(workgroupCount),
                                                                 vk::BufferUsageFlagBits::eStorageBuffer,
                                                                 memoryMapping);
                buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                                   vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
                buffers.partitionIndices = alloc->createBuffer(PartitionIndicesLayout::size(weightCount),
                                                               vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
                buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
                                                              vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
                buffers.partitionDecoupledState = alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
                                                                      vk::BufferUsageFlagBits::eStorageBuffer,
                                                                      memoryMapping);
                buffers.splits = alloc->createBuffer(SplitLayout::size(splitCount),
                                                     vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
                buffers.aliasTable = alloc->createBuffer(AliasTableLayout::size(weightCount),
                                                         vk::BufferUsageFlagBits::eStorageBuffer
                                                         | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            } else {
                buffers.weights = alloc->createBuffer(WeightsLayout::size(weightCount),
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      memoryMapping);
                buffers.meanDecoupledStates = alloc->createBuffer(MeanDecoupledStatesLayout::size(workgroupCount),
                                                                 {}, memoryMapping);
                buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                                   {}, memoryMapping);
                buffers.partitionIndices = alloc->createBuffer(PartitionIndicesLayout::size(weightCount),
                                                               {}, memoryMapping);
                buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
                                                              {}, memoryMapping);
                buffers.partitionDecoupledState = alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
                                                                      {}, memoryMapping);
                buffers.splits = alloc->createBuffer(SplitLayout::size(splitCount),
                                                     {}, memoryMapping);
                buffers.aliasTable = alloc->createBuffer(AliasTableLayout::size(weightCount),
                                                         vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }
            return buffers;
        }
    };


    class ScalarPsa {

    public:
        using Buffers = ScalarPsaBuffers;
        using weight_t = glsl::float_t;
        static constexpr uint32_t DEFAULT_WORKGROUP_SIZE = Buffers::DEFAULT_WORKGROUP_SIZE;
        static constexpr uint32_t DEFAULT_ROWS = Buffers::DEFAULT_ROWS;

        explicit ScalarPsa(const merian::ContextHandle &context) : m_mean(context, DEFAULT_WORKGROUP_SIZE, DEFAULT_ROWS, false),
                                                                   m_prefixPartition(context, DEFAULT_WORKGROUP_SIZE, DEFAULT_ROWS, true, true),
                                                                   m_split(context),
                                                                   m_pack(context) {
        }

        void run(const vk::CommandBuffer cmd, const Buffers &buffers, const std::size_t weightCount,
            const std::size_t splitCount, std::optional<merian::ProfilerHandle> profiler = std::nullopt) {

            DecoupledMeanBuffers meanBuffers;
            meanBuffers.elements = buffers.weights;
            meanBuffers.decoupledStates = buffers.meanDecoupledStates;
            meanBuffers.mean = buffers.mean;
            if (profiler.has_value()) {
                MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Substep: Mean");
                m_mean.run(cmd, meanBuffers, weightCount);
            }else {
                m_mean.run(cmd, meanBuffers, weightCount);
            }

            common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd,
                buffers.mean);

            DecoupledPrefixPartitionBuffers partitionBuffers;
            partitionBuffers.elements = buffers.weights;
            partitionBuffers.pivot = buffers.mean;
            partitionBuffers.partition = buffers.partitionIndices;
            partitionBuffers.partitionPrefix = buffers.partitionPrefix;
            partitionBuffers.batchDescriptors = buffers.partitionDecoupledState;

            if (profiler.has_value()) {
                MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Substep: PrefixPartition");
                m_prefixPartition.run(cmd, partitionBuffers, weightCount);
            }else {
                m_prefixPartition.run(cmd, partitionBuffers, weightCount);
            }

            common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd,
                buffers.partitionIndices);
            common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd,
                buffers.partitionPrefix);

            ScalarSplitBuffers splitBuffers;
            splitBuffers.mean = buffers.mean;
            splitBuffers.splits = buffers.splits;
            splitBuffers.partitionPrefix = buffers.partitionPrefix;

            if (profiler.has_value()) {
                MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Substep: Split");
                m_split.run(cmd, splitBuffers, weightCount, splitCount);
            }else {
                m_split.run(cmd, splitBuffers, weightCount, splitCount);
            }


            common_vulkan::pipelineBarrierComputeReadAfterComputeWrite(cmd,
                buffers.splits);

            ScalarPackBuffers packBuffers;
            packBuffers.weights = buffers.weights;
            packBuffers.mean = buffers.mean;
            packBuffers.splits = buffers.splits;
            packBuffers.partitionIndices = buffers.partitionIndices;
            packBuffers.aliasTable = buffers.aliasTable;

            if (profiler.has_value()) {
                MERIAN_PROFILE_SCOPE_GPU(*profiler, cmd, "Substep: Pack");
                m_pack.run(cmd, weightCount, splitCount, packBuffers);
            }else {
                m_pack.run(cmd, weightCount, splitCount, packBuffers);
            }
        }

    private:
        DecoupledMean<weight_t> m_mean;
        DecoupledPrefixPartition<weight_t> m_prefixPartition;
        ScalarSplit<weight_t> m_split;
        ScalarPack<weight_t> m_pack;
    };
}
