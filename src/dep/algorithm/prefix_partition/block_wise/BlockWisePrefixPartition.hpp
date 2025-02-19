#pragma once

#include "./compatible.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/prefix_partition/block_wise/block_reduce/PrefixPartitionBlockReduce.hpp"
#include "src/wrs/algorithm/prefix_partition/block_wise/block_scan/PrefixPartitionBlockScan.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace wrs {

class BlockWisePrefixPartitionBuffers {
  public:
    using Self = BlockWisePrefixPartitionBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    merian::BufferHandle elements;
    template <block_wise_prefix_partition_compatible T>
    using ElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using ElementsView = layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <block_wise_prefix_partition_compatible T>
    using PivotLayout = layout::PrimitiveLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T> //
    using PivotView = layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <block_wise_prefix_partition_compatible T>
    using PartitionElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using PartitionElementsView = layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle partitionPrefix;
    template <block_wise_prefix_partition_compatible T>
    using PartitionPrefixLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout<T>>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = layout::PrimitiveLayout<glsl::uint, storageQualifier>;
    using HeavyCountView = layout::BufferView<HeavyCountLayout>;

    merian::BufferHandle blockHeavyCount;
    using BlockHeavyCountLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using BlockHeavyCountView = layout::BufferView<BlockHeavyCountLayout>;

    merian::BufferHandle blockHeavyReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsView = layout::BufferView<BlockHeavyReductionsLayout<T>>;

    merian::BufferHandle blockLightReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsView = layout::BufferView<BlockLightReductionsLayout<T>>;

    template <block_wise_prefix_partition_compatible T>
    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N,
                         glsl::uint blockCount) {
        Self buffers;

        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.elements = alloc->createBuffer(ElementsLayout<T>::size(N),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferDst,
                                                   memoryMapping);
            buffers.pivot = alloc->createBuffer(PivotLayout<T>::size(),
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                memoryMapping);
            buffers.partitionIndices = alloc->createBuffer(
                PartitionIndicesLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                memoryMapping);
            buffers.partitionElements = alloc->createBuffer(
                PartitionElementsLayout<T>::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                memoryMapping);
            buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                                          vk::BufferUsageFlagBits::eStorageBuffer |
                                                              vk::BufferUsageFlagBits::eTransferSrc,
                                                          memoryMapping);
            buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                     memoryMapping);
            buffers.blockHeavyCount =
                alloc->createBuffer(BlockHeavyCountLayout::size(blockCount),
                                    vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            buffers.blockHeavyReductions =
                alloc->createBuffer(BlockHeavyReductionsLayout<T>::size(blockCount),
                                    vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
            buffers.blockLightReductions =
                alloc->createBuffer(BlockLightReductionsLayout<T>::size(blockCount),
                                    vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
        } else {
            buffers.elements = alloc->createBuffer(
                ElementsLayout<T>::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.pivot = alloc->createBuffer(
                PivotLayout<T>::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.partitionIndices =
                alloc->createBuffer(PartitionIndicesLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.partitionElements =
                alloc->createBuffer(PartitionElementsLayout<T>::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.partitionPrefix =
                alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.heavyCount = alloc->createBuffer(
                HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.blockHeavyCount = nullptr;
            buffers.blockHeavyReductions = nullptr;
            buffers.blockLightReductions = nullptr;
        }
        return buffers;
    }
};

struct BlockWisePrefixPartitionConfig {
    const BlockWisePrefixPartitionBlockReduceConfig reduceConfig;
    const BlockScanConfig blockScanConfig;
    const BlockWisePrefixPartitionBlockScanConfig scanConfig;

    constexpr explicit BlockWisePrefixPartitionConfig(
        BlockWisePrefixPartitionBlockReduceConfig reduceConfig,
        BlockScanConfig blockScanConfig,
        BlockWisePrefixPartitionBlockScanConfig scanConfig)
        : reduceConfig(reduceConfig), blockScanConfig(blockScanConfig), scanConfig(scanConfig) {}

    constexpr explicit BlockWisePrefixPartitionConfig(glsl::uint workgroupSize,
                                                      glsl::uint rows,
                                                      BlockScanVariant variant,
                                                      glsl::uint sequentialScanLength = 1)
        : reduceConfig(BlockWisePrefixPartitionBlockReduceConfig(workgroupSize, rows)),
          blockScanConfig(
              512, 8, BlockScanVariant::RANKED_STRIDED | BlockScanVariant::EXCLUSIVE, 1, true),
          scanConfig(workgroupSize, rows, sequentialScanLength, variant) {}

    inline glsl::uint blockSize() const {
        return reduceConfig.blockSize();
    }
};

template <block_wise_prefix_partition_compatible T> class BlockWisePrefixPartition {
  public:
    using Buffers = BlockWisePrefixPartitionBuffers;
    using Config = BlockWisePrefixPartitionConfig;

    BlockWisePrefixPartition(const merian::ContextHandle& context,
                             const merian::ShaderCompilerHandle& shaderCompiler,
                             BlockWisePrefixPartitionConfig config)
        : m_reduce(context, shaderCompiler, config.reduceConfig),
          m_countBlockScan(context,
                           shaderCompiler,
                           BlockScanConfig(config.blockScanConfig.workgroupSize,
                                           config.blockScanConfig.rows,
                                           config.blockScanConfig.variant,
                                           config.blockScanConfig.sequentialScanLength,
                                           true)),
          m_heavyBlockScan(context,
                           shaderCompiler,
                           BlockScanConfig(config.blockScanConfig.workgroupSize,
                                           config.blockScanConfig.rows,
                                           config.blockScanConfig.variant,
                                           config.blockScanConfig.sequentialScanLength,
                                           false)),
          m_lightBlockScan(context,
                           shaderCompiler,
                           BlockScanConfig(config.blockScanConfig.workgroupSize,
                                           config.blockScanConfig.rows,
                                           config.blockScanConfig.variant,
                                           config.blockScanConfig.sequentialScanLength,
                                           false)),
          m_scan(context, shaderCompiler, config.scanConfig) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {

        typename BlockWisePrefixPartitionBlockReduce<T>::Buffers reduceBuffers;
        reduceBuffers.elements = buffers.elements;
        reduceBuffers.pivot = buffers.pivot;
        reduceBuffers.blockHeavyCount = buffers.blockHeavyCount;
        reduceBuffers.blockHeavyReductions = buffers.blockHeavyReductions;
        reduceBuffers.blockLightReductions = buffers.blockLightReductions;

        if (profiler.has_value()) {
            profiler.value()->start("Reduce");
            profiler.value()->cmd_start(cmd, "Reduce");
        }

        m_reduce.run(cmd, reduceBuffers, N);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {buffers.blockHeavyCount->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead),
                      buffers.blockHeavyReductions->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                   vk::AccessFlagBits::eShaderRead),
                      buffers.blockLightReductions->buffer_barrier(
                          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead)});

        if (profiler.has_value()) {
            profiler.value()->start("BlockScan");
            profiler.value()->cmd_start(cmd, "BlockScan");
        }

        glsl::uint blockCount = (N + m_reduce.blockSize() - 1) / m_reduce.blockSize();

        BlockScan<glsl::uint>::Buffers countBlockScanBuffers;
        countBlockScanBuffers.elements = buffers.blockHeavyCount;
        countBlockScanBuffers.prefixSum = buffers.blockHeavyCount;
        countBlockScanBuffers.reductions = buffers.heavyCount;

        m_countBlockScan.run(cmd, countBlockScanBuffers, blockCount);

        typename BlockScan<T>::Buffers heavyBlockScanBuffers;
        heavyBlockScanBuffers.elements = buffers.blockHeavyReductions;
        heavyBlockScanBuffers.prefixSum = buffers.blockHeavyReductions;

        m_heavyBlockScan.run(cmd, heavyBlockScanBuffers, blockCount);

        typename BlockScan<T>::Buffers lightBlockScanBuffers;
        lightBlockScanBuffers.elements = buffers.blockLightReductions;
        lightBlockScanBuffers.prefixSum = buffers.blockLightReductions;

        m_lightBlockScan.run(cmd, lightBlockScanBuffers, blockCount);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {buffers.blockHeavyCount->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead),
                      buffers.blockHeavyReductions->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                   vk::AccessFlagBits::eShaderRead),
                      buffers.blockLightReductions->buffer_barrier(
                          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead)});

        typename BlockWisePrefixPartitionBlockScan<T>::Buffers scanBuffers;
        scanBuffers.elements = buffers.elements;
        scanBuffers.pivot = buffers.pivot;
        scanBuffers.blockHeavyCount = buffers.blockHeavyCount;
        scanBuffers.blockHeavyReductions = buffers.blockHeavyReductions;
        scanBuffers.blockLightReductions = buffers.blockLightReductions;
        scanBuffers.partitionIndices = buffers.partitionIndices;
        scanBuffers.partitionElements = buffers.partitionElements;
        scanBuffers.partitionPrefix = buffers.partitionPrefix;

        if (profiler.has_value()) {
            profiler.value()->start("PrefixPartitionScan");
            profiler.value()->cmd_start(cmd, "PrefixPartitionScan");
        }
        m_scan.run(cmd, scanBuffers, N);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

  private:
    BlockWisePrefixPartitionBlockReduce<T> m_reduce;
    BlockScan<glsl::uint> m_countBlockScan;
    BlockScan<T> m_heavyBlockScan;
    BlockScan<T> m_lightBlockScan;
    BlockWisePrefixPartitionBlockScan<T> m_scan;
};

} // namespace wrs
