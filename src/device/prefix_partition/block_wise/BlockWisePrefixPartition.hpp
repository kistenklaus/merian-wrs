#pragma once

#include "./compatible.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_partition/PrefixPartitionAllocFlags.hpp"
#include "src/device/prefix_partition/block_wise/block_reduce/PrefixPartitionBlockReduce.hpp"
#include "src/device/prefix_partition/block_wise/block_scan/PrefixPartitionBlockScan.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device {

class BlockWisePrefixPartitionBuffers {
  public:
    using Self = BlockWisePrefixPartitionBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    merian::BufferHandle elements;
    template <block_wise_prefix_partition_compatible T>
    using ElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using ElementsView = host::layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <block_wise_prefix_partition_compatible T>
    using PivotLayout = host::layout::PrimitiveLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T> //
    using PivotView = host::layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <block_wise_prefix_partition_compatible T>
    using PartitionElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using PartitionElementsView = host::layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle partitionPrefix;
    template <block_wise_prefix_partition_compatible T>
    using PartitionPrefixLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout<T>>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    merian::BufferHandle blockHeavyCount;
    using BlockHeavyCountLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using BlockHeavyCountView = host::layout::BufferView<BlockHeavyCountLayout>;

    merian::BufferHandle blockHeavyReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockHeavyReductionsView = host::layout::BufferView<BlockHeavyReductionsLayout<T>>;

    merian::BufferHandle blockLightReductions;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <block_wise_prefix_partition_compatible T>
    using BlockLightReductionsView = host::layout::BufferView<BlockLightReductionsLayout<T>>;

    template <block_wise_prefix_partition_compatible T>
    static Self
    allocate(const merian::ResourceAllocatorHandle& alloc,
             merian::MemoryMappingType memoryMapping,
             host::glsl::uint N,
             host::glsl::uint blockCount,
             PrefixPartitionAllocFlags allocFlags = PrefixPartitionAllocFlags::ALLOC_ALL) {
        Self buffers;

        if (memoryMapping == merian::MemoryMappingType::NONE) {
            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements = alloc->createBuffer(ElementsLayout<T>::size(N),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       memoryMapping);
            }
            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PIVOT) != 0) {
                buffers.pivot = alloc->createBuffer(PivotLayout<T>::size(),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferDst,
                                                    memoryMapping);
            }
            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
                buffers.partitionIndices = alloc->createBuffer(
                    PartitionIndicesLayout::size(N),
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                    memoryMapping);
            }
            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
                buffers.partitionElements = alloc->createBuffer(
                    PartitionElementsLayout<T>::size(N),
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                    memoryMapping);
            }
            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_PREFIX) != 0) {
                buffers.partitionPrefix = alloc->createBuffer(
                    PartitionPrefixLayout<T>::size(N),
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                    memoryMapping);
            }
            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
                buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                         vk::BufferUsageFlagBits::eStorageBuffer |
                                                             vk::BufferUsageFlagBits::eTransferSrc,
                                                         memoryMapping);
            }
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

            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements =
                    alloc->createBuffer(ElementsLayout<T>::size(N),
                                        vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }

            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PIVOT) != 0) {
                buffers.pivot = alloc->createBuffer(
                    PivotLayout<T>::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }

            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
                buffers.partitionIndices =
                    alloc->createBuffer(PartitionIndicesLayout::size(N),
                                        vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }

            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
                buffers.partitionElements =
                    alloc->createBuffer(PartitionElementsLayout<T>::size(N),
                                        vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }

            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_PREFIX) != 0) {
                buffers.partitionPrefix =
                    alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                        vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }

            if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
                buffers.heavyCount = alloc->createBuffer(
                    HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }

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

    constexpr explicit BlockWisePrefixPartitionConfig(host::glsl::uint workgroupSize,
                                                      host::glsl::uint rows,
                                                      BlockScanVariant variant)
        : reduceConfig(workgroupSize, rows),
          blockScanConfig(512, 8, BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE, 1, true),
          scanConfig(workgroupSize, rows, 1, variant) {}

    inline host::glsl::uint blockSize() const {
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
          m_scan(context, shaderCompiler, config.scanConfig) {
        assert((config.blockScanConfig.variant & BlockScanVariant::EXCLUSIVE) != 0);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("Block-Wise-Prefix-Partition");
            profiler.value()->cmd_start(cmd, "Block-Wise-Prefix-Partition");
        }
#endif

        typename BlockWisePrefixPartitionBlockReduce<T>::Buffers reduceBuffers;
        reduceBuffers.elements = buffers.elements;
        reduceBuffers.pivot = buffers.pivot;
        reduceBuffers.blockHeavyCount = buffers.blockHeavyCount;
        reduceBuffers.blockHeavyReductions = buffers.blockHeavyReductions;
        reduceBuffers.blockLightReductions = buffers.blockLightReductions;

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("Reduce");
            profiler.value()->cmd_start(cmd, "Reduce");
        }
#endif

        m_reduce.run(cmd, reduceBuffers, N);

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {buffers.blockHeavyCount->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead),
                      buffers.blockHeavyReductions->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                   vk::AccessFlagBits::eShaderRead),
                      buffers.blockLightReductions->buffer_barrier(
                          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead)});

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("BlockScan");
            profiler.value()->cmd_start(cmd, "BlockScan");
        }
#endif

        host::glsl::uint blockCount = (N + m_reduce.blockSize() - 1) / m_reduce.blockSize();

        BlockScan<host::glsl::uint>::Buffers countBlockScanBuffers;
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

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif

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

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("PrefixPartitionScan");
            profiler.value()->cmd_start(cmd, "PrefixPartitionScan");
        }
#endif
        m_scan.run(cmd, scanBuffers, N);
#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif
    }

  private:
    BlockWisePrefixPartitionBlockReduce<T> m_reduce;
    BlockScan<host::glsl::uint> m_countBlockScan;
    BlockScan<T> m_heavyBlockScan;
    BlockScan<T> m_lightBlockScan;
    BlockWisePrefixPartitionBlockScan<T> m_scan;
};

} // namespace device
