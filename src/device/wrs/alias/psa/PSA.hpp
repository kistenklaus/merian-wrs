#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "src/device/mean/Mean.hpp"
#include "src/device/mean/MeanAllocFlags.hpp"
#include "src/device/prefix_partition/PrefixPartition.hpp"
#include "src/device/prefix_partition/PrefixPartitionAllocFlags.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/device/wrs/alias/psa/layout/split.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPack.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPackAllocFlags.hpp"
#include "src/host/types/glsl.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <cassert>
#include <fmt/base.h>

namespace device {

struct PSAConfig {
    const MeanConfig meanConfig;
    const PrefixPartitionConfig prefixPartitionConfig;
    const SplitPackConfig splitPackConfig;
    const bool usePartitionElements;

    constexpr explicit PSAConfig(MeanConfig meanConfig,
                                 PrefixPartitionConfig prefixPartitionConfig,
                                 SplitPackConfig splitPackConfig,
                                 bool usePartitionElements)
        : meanConfig(meanConfig), prefixPartitionConfig(prefixPartitionConfig),
          splitPackConfig(splitPackConfig), usePartitionElements(usePartitionElements) {}

    constexpr host::glsl::uint splitSize() const {
        return splitPackConfigSplitSize(splitPackConfig);
    }

    inline std::string name() const {
      return fmt::format("PSA-[{}]-[{}]-[{}]", meanConfigName(meanConfig),
          prefixPartitionConfigName(prefixPartitionConfig),
          splitPackConfigName(splitPackConfig));
    }
};

struct PSABuffers {
    using Self = PSABuffers;
    using weight_type = host::glsl::f32;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableLayout = device::details::AliasTableLayout;
    using AliasTableView = host::layout::BufferView<AliasTableLayout>;

    merian::BufferHandle m_mean;
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle m_heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    merian::BufferHandle m_partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle m_partitionElements;
    using PartitionElementsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using PartitionElementsView = host::layout::BufferView<PartitionElementsLayout>;

    merian::BufferHandle m_partitionPrefix;
    using PartitionPrefixLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout>;

    Mean<weight_type>::Buffers::Internals m_meanInternals;
    PrefixPartition<weight_type>::Buffers::Internals m_prefixPartitionInternals;
    SplitPack::Buffers::Internals m_splitPackInternals;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         const PSAConfig config,
                         host::glsl::uint N,
                         // makes internal buffers readable over staging buffers
                         // however might also reduce performance marginaly.
                         bool leakInternals = false) {
        Self buffers;
        Mean<weight_type>::Buffers meanBuffers = Mean<weight_type>::Buffers::allocate<weight_type>(
            alloc, memoryMapping, config.meanConfig, N, MeanAllocFlags::ALLOC_ONLY_INTERNALS);
        buffers.m_meanInternals = meanBuffers.m_internalBuffers;

        PrefixPartition<weight_type>::Buffers prefixPartitionBuffers =
            PrefixPartition<weight_type>::Buffers::allocate<weight_type>(
                alloc, memoryMapping, config.prefixPartitionConfig, N,
                PrefixPartitionAllocFlags::ALLOC_ONLY_INTERNALS);
        buffers.m_prefixPartitionInternals = prefixPartitionBuffers.m_internalBuffers;

        SplitPack::Buffers splitPackBuffers =
            SplitPack::Buffers::allocate(alloc, memoryMapping, config.splitPackConfig, N,
                                         SplitPackAllocFlags::ALLOC_ONLY_INTERNALS);
        buffers.m_splitPackInternals = splitPackBuffers.m_internals;

        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                  memoryMapping, "psa-weights");

            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, N);

        } else {
            buffers.weights =
                alloc->createBuffer(WeightsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc,
                                    memoryMapping, "psa-weights");
            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping, vk::BufferUsageFlagBits::eTransferDst, N);
        }

        vk::BufferUsageFlags internalUsage = vk::BufferUsageFlagBits::eStorageBuffer;
        if (leakInternals) {
            if (memoryMapping == merian::MemoryMappingType::NONE) {
                internalUsage |= vk::BufferUsageFlagBits::eTransferSrc;
            } else {
                internalUsage |= vk::BufferUsageFlagBits::eTransferDst;
            }
        }
        buffers.m_mean =
            alloc->createBuffer(MeanLayout::size(), internalUsage, memoryMapping, "psa-mean");
        buffers.m_heavyCount = alloc->createBuffer(HeavyCountLayout::size(), internalUsage,
                                                   memoryMapping, "psa-heavy-count");
        buffers.m_partitionIndices = alloc->createBuffer(
            PartitionIndicesLayout::size(N), internalUsage, memoryMapping, "psa-partition-indices");
        if (config.usePartitionElements) {
            buffers.m_partitionElements =
                alloc->createBuffer(PartitionElementsLayout::size(N), internalUsage, memoryMapping,
                                    "psa-partition-elements");
        } else {
            buffers.m_partitionElements = nullptr;
        }
        buffers.m_partitionPrefix = alloc->createBuffer(
            PartitionPrefixLayout::size(N), internalUsage, memoryMapping, "psa-partition-prefix");

        return buffers;
    }
};

class PSA {
  public:
    using Buffers = PSABuffers;
    using Config = PSAConfig;
    using weight_type = host::glsl::f32;

  private:
    using MeanBuffers = Mean<weight_type>::Buffers;
    using PrefixPartitionBuffers = PrefixPartition<weight_type>::Buffers;
    using SplitPackBuffers = SplitPack::Buffers;

  public:
    explicit PSA(const merian::ContextHandle& context,
                 const merian::ShaderCompilerHandle& shaderCompiler,
                 const Config& config)
        : m_mean(context, shaderCompiler, config.meanConfig),
          m_prefixPartition(
              context, shaderCompiler, config.prefixPartitionConfig, config.usePartitionElements),
          m_splitPack(context, shaderCompiler, config.splitPackConfig, config.usePartitionElements),
          m_usePartitionElements(config.usePartitionElements) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             const std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {

        {
            MeanBuffers meanBuffers;
            meanBuffers.elements = buffers.weights;
            meanBuffers.mean = buffers.m_mean;
            meanBuffers.m_internalBuffers = buffers.m_meanInternals;

            m_mean.run(cmd, meanBuffers, N, profiler);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.m_mean->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                    vk::AccessFlagBits::eShaderRead));

        PrefixPartitionBuffers prefixPartitionBuffers;
        prefixPartitionBuffers.elements = buffers.weights;
        prefixPartitionBuffers.pivot = buffers.m_mean;
        prefixPartitionBuffers.partitionIndices = buffers.m_partitionIndices;
        prefixPartitionBuffers.partitionElements = buffers.m_partitionElements;
        prefixPartitionBuffers.partitionPrefix = buffers.m_partitionPrefix;
        prefixPartitionBuffers.heavyCount = buffers.m_heavyCount;
        prefixPartitionBuffers.m_internalBuffers = buffers.m_prefixPartitionInternals;

        m_prefixPartition.run(cmd, prefixPartitionBuffers, N, profiler);

        if (m_usePartitionElements) {
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eComputeShader,
                         {
                             buffers.m_heavyCount->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                  vk::AccessFlagBits::eShaderRead),
                             buffers.m_partitionIndices->buffer_barrier(
                                 vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                             buffers.m_partitionPrefix->buffer_barrier(
                                 vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                             buffers.m_partitionElements->buffer_barrier(
                                 vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                         });
        } else {
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eComputeShader,
                         {
                             buffers.m_heavyCount->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                  vk::AccessFlagBits::eShaderRead),
                             buffers.m_partitionIndices->buffer_barrier(
                                 vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                             buffers.m_partitionPrefix->buffer_barrier(
                                 vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                         });
        }

        SplitPackBuffers splitPackBuffers;
        splitPackBuffers.heavyCount = buffers.m_heavyCount;
        splitPackBuffers.weights = buffers.weights;
        splitPackBuffers.partitionElements = buffers.m_partitionElements;
        splitPackBuffers.partitionIndices = buffers.m_partitionIndices;
        splitPackBuffers.partitionPrefix = buffers.m_partitionPrefix;
        splitPackBuffers.mean = buffers.m_mean;
        splitPackBuffers.aliasTable = buffers.aliasTable;
        splitPackBuffers.m_internals = buffers.m_splitPackInternals;

        m_splitPack.run(cmd, splitPackBuffers, N, profiler);
    }

  private:
    Mean<weight_type> m_mean;
    PrefixPartition<weight_type> m_prefixPartition;
    SplitPack m_splitPack;
    const bool m_usePartitionElements;
};

} // namespace device
