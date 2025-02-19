#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/partition/PartitionAllocFlags.hpp"
#include "src/device/partition/block_wise/block_scan/PartitionBlockScan.hpp"
#include "src/device/partition/block_wise/combine/PartitionCombine.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/layout/PrimitiveLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>
namespace device {

struct BlockWisePartitionBuffers {
    using Self = BlockWisePartitionBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = host::layout::BufferView<ElementsLayout>;

    merian::BufferHandle pivot;
    using PivotLayout = host::layout::PrimitiveLayout<float, storageQualifier>;
    using PivotView = host::layout::BufferView<PivotLayout>;

    merian::BufferHandle indices;
    using IndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using IndicesView = host::layout::BufferView<IndicesLayout>;

    merian::BufferHandle blockIndices;
    using BlockIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using BlockIndicesView = host::layout::BufferView<BlockIndicesLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partition;
    using PartitionLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using PartitionView = host::layout::BufferView<PartitionLayout>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N,
                         host::glsl::uint blockCount,
                         PartitionAllocFlags allocFlags = PartitionAllocFlags::ALLOC_ALL) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            if ((allocFlags & PartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       memoryMapping);
            }
            if ((allocFlags & PartitionAllocFlags::ALLOC_PIVOT) != 0) {
                buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferDst,
                                                    memoryMapping);
            }

            buffers.indices = alloc->createBuffer(IndicesLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);

            buffers.blockIndices = alloc->createBuffer(BlockIndicesLayout::size(blockCount),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferSrc,
                                                       memoryMapping);

            if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
                buffers.partitionIndices = alloc->createBuffer(
                    PartitionIndicesLayout::size(N),
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                    memoryMapping);
            }

            if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
                buffers.partition = alloc->createBuffer(PartitionLayout::size(N),
                                                        vk::BufferUsageFlagBits::eStorageBuffer |
                                                            vk::BufferUsageFlagBits::eTransferSrc,
                                                        memoryMapping);
            }

            if ((allocFlags & PartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
                buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                         vk::BufferUsageFlagBits::eStorageBuffer |
                                                             vk::BufferUsageFlagBits::eTransferSrc,
                                                         memoryMapping);
            }
        } else {

            if ((allocFlags & PartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements = alloc->createBuffer(
                    ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }

            if ((allocFlags & PartitionAllocFlags::ALLOC_PIVOT) != 0) {
                buffers.pivot = alloc->createBuffer(
                    PivotLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }

            buffers.indices = alloc->createBuffer(
                IndicesLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

            buffers.blockIndices =
                alloc->createBuffer(BlockIndicesLayout::size(blockCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

            if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
                buffers.partitionIndices =
                    alloc->createBuffer(PartitionIndicesLayout::size(N),
                                        vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }

            if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
                buffers.partition = alloc->createBuffer(
                    PartitionLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }

            if ((allocFlags & PartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
                buffers.heavyCount = alloc->createBuffer(
                    HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }
        }
        return buffers;
    }
};

struct BlockWisePartitionConfig {
    const PartitionBlockScanConfig elementScanConfig;
    const BlockScanConfig blockScanConfig;
    const PartitionCombineConfig blockCombineConfig;

    constexpr explicit BlockWisePartitionConfig(host::glsl::uint workgroupSize,
                                                host::glsl::uint rows,
                                                host::glsl::uint sequentialScanLength,
                                                BlockScanVariant scanVariant)
        : elementScanConfig(
              workgroupSize, rows, sequentialScanLength, scanVariant | BlockScanVariant::EXCLUSIVE),
          blockScanConfig(
              512, 8, BlockScanVariant::RANKED_STRIDED | BlockScanVariant::EXCLUSIVE, 1, true),
          blockCombineConfig(workgroupSize, rows * sequentialScanLength, 1, 1) {
        // Currently not implemented
        assert((scanVariant & BlockScanVariant::RANKED_STRIDED) !=
               BlockScanVariant::RANKED_STRIDED);
    }

    constexpr explicit BlockWisePartitionConfig(PartitionBlockScanConfig elementScanConfig,
                                                BlockScanConfig blockScanConfig,
                                                PartitionCombineConfig combineConfig)
        : elementScanConfig(elementScanConfig), blockScanConfig(blockScanConfig),
          blockCombineConfig(combineConfig) {
        if (!((elementScanConfig.variant & BlockScanVariant::EXCLUSIVE) ==
              BlockScanVariant::EXCLUSIVE)) {
            throw std::runtime_error(
                "ElementScan of BlockWisePartition is required to be EXCLUSIVE");
        }
        if (!(elementScanConfig.blockSize() == blockCombineConfig.blockSize())) {
            throw std::runtime_error(
                "The blockSize of elementScan must match the blockSize of blockCombine");
        }
        if (!((blockScanConfig.variant & BlockScanVariant::EXCLUSIVE) ==
              BlockScanVariant::EXCLUSIVE)) {
            throw std::runtime_error("BlockScan of BlockWisePartition is required to be EXCLUSIVE");
        }
    }

    inline host::glsl::uint blockSize() const {
        return elementScanConfig.blockSize();
    }

    inline host::glsl::uint maxElementCount() const {
        return blockScanConfig.blockSize() * elementScanConfig.blockSize();
    }
};

struct BlockWisePartition {
  public:
    using Buffers = BlockWisePartitionBuffers;
    using Config = BlockWisePartitionConfig;

    BlockWisePartition(const merian::ContextHandle& context,
                       const merian::ShaderCompilerHandle& shaderCompiler,
                       BlockWisePartitionConfig config)
        : m_elementScan(context, shaderCompiler, config.elementScanConfig),
          m_blockScan(context, shaderCompiler, config.blockScanConfig),
          m_combine(context, shaderCompiler, config.blockCombineConfig) {}

    void run(const merian::CommandBufferHandle& cmd,
             const BlockWisePartitionBuffers& buffers,
             host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        assert(N <= m_blockScan.blockSize() * m_elementScan.blockSize());

        PartitionBlockScan::Buffers elementScanBuffers;
        elementScanBuffers.elements = buffers.elements;
        elementScanBuffers.pivot = buffers.pivot;
        elementScanBuffers.indices = buffers.indices;
        elementScanBuffers.blockCount = buffers.blockIndices;

        host::glsl::uint blockCount =
            (N + m_elementScan.blockSize() - 1) / m_elementScan.blockSize();
        if (profiler.has_value()) {

            profiler.value()->start(fmt::format("ElementScan {}", blockCount));
            profiler.value()->cmd_start(cmd, fmt::format("ElementScan {}", blockCount));
        }
        m_elementScan.run(cmd, elementScanBuffers, N);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.blockIndices->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                          vk::AccessFlagBits::eShaderRead));

        BlockScan<host::glsl::uint>::Buffers blockScanBuffers;
        blockScanBuffers.elements = buffers.blockIndices;
        blockScanBuffers.prefixSum = buffers.blockIndices;
        blockScanBuffers.reductions = buffers.heavyCount;

        if (profiler.has_value()) {
            profiler.value()->start("BlockScan");
            profiler.value()->cmd_start(cmd, "BlockScan");
        }
        m_blockScan.run(cmd, blockScanBuffers, blockCount);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.blockIndices->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                          vk::AccessFlagBits::eShaderRead));

        PartitionCombine::Buffers combineBuffers;
        combineBuffers.elements = buffers.elements;
        combineBuffers.pivot = buffers.pivot;
        combineBuffers.indices = buffers.indices;
        combineBuffers.blockIndices = buffers.blockIndices;
        combineBuffers.partitionIndices = buffers.partitionIndices;
        combineBuffers.partition = buffers.partition;

        if (profiler.has_value()) {
            host::glsl::uint count = (N + m_combine.tileSize() - 1) / m_combine.tileSize();
            profiler.value()->start(fmt::format("Combine {}", count));
            profiler.value()->cmd_start(cmd, fmt::format("Combine {}", count));
        }
        m_combine.run(cmd, combineBuffers, N);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

    inline host::glsl::uint maxElementCount() const {
        return m_elementScan.blockSize() * m_blockScan.blockSize();
    }

  private:
    PartitionBlockScan m_elementScan;
    BlockScan<host::glsl::uint> m_blockScan;
    PartitionCombine m_combine;
};

} // namespace device
