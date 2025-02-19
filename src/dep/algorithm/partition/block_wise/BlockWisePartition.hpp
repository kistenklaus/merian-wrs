#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/partition/block_wise/block_scan/PartitionBlockScan.hpp"
#include "src/wrs/algorithm/partition/block_wise/combine/PartitionCombine.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>
namespace wrs {

struct BlockWisePartitionBuffers {
    using Self = BlockWisePartitionBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    merian::BufferHandle pivot;
    using PivotLayout = layout::PrimitiveLayout<float, storageQualifier>;
    using PivotView = layout::BufferView<PivotLayout>;

    merian::BufferHandle indices;
    using IndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using IndicesView = layout::BufferView<IndicesLayout>;

    merian::BufferHandle blockIndices;
    using BlockIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using BlockIndicesView = layout::BufferView<BlockIndicesLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partition;
    using PartitionLayout = layout::ArrayLayout<float, storageQualifier>;
    using PartitionView = layout::BufferView<PartitionLayout>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = layout::PrimitiveLayout<glsl::uint, storageQualifier>;
    using HeavyCountView = layout::BufferView<HeavyCountLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         glsl::uint N,
                         glsl::uint blockCount) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferDst,
                                                   memoryMapping);
            buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                memoryMapping);

            buffers.indices = alloc->createBuffer(IndicesLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);
            buffers.blockIndices = alloc->createBuffer(BlockIndicesLayout::size(blockCount),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferSrc,
                                                       memoryMapping);

            buffers.partitionIndices = alloc->createBuffer(
                PartitionIndicesLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                memoryMapping);

            buffers.partition = alloc->createBuffer(PartitionLayout::size(N),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferSrc,
                                                    memoryMapping);

            buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                     memoryMapping);

        } else {
            buffers.elements = alloc->createBuffer(
                ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);

            buffers.pivot = alloc->createBuffer(
                PivotLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);

            buffers.indices = alloc->createBuffer(
                IndicesLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

            buffers.blockIndices =
                alloc->createBuffer(BlockIndicesLayout::size(blockCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

            buffers.partitionIndices =
                alloc->createBuffer(PartitionIndicesLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

            buffers.partition = alloc->createBuffer(
                PartitionLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

            buffers.heavyCount = alloc->createBuffer(
                HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

struct BlockWisePartitionConfig {
    const PartitionBlockScanConfig elementScanConfig;
    const BlockScanConfig blockScanConfig;
    const PartitionCombineConfig blockCombineConfig;

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

    inline glsl::uint blockSize() const {
        return elementScanConfig.blockSize();
    }

    inline glsl::uint maxElementCount() const {
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
             glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        assert(N <= m_blockScan.blockSize() * m_elementScan.blockSize());

        PartitionBlockScan::Buffers elementScanBuffers;
        elementScanBuffers.elements = buffers.elements;
        elementScanBuffers.pivot = buffers.pivot;
        elementScanBuffers.indices = buffers.indices;
        elementScanBuffers.blockCount = buffers.blockIndices;

        glsl::uint blockCount = (N + m_elementScan.blockSize() - 1) / m_elementScan.blockSize();
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

        BlockScan<glsl::uint>::Buffers blockScanBuffers;
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
            glsl::uint count = (N + m_combine.tileSize() - 1) / m_combine.tileSize();
            profiler.value()->start(fmt::format("Combine {}", count));
            profiler.value()->cmd_start(cmd, fmt::format("Combine {}", count));
        }
        m_combine.run(cmd, combineBuffers, N);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

    inline glsl::uint maxElementCount() const {
        return m_elementScan.blockSize() * m_blockScan.blockSize();
    }

  private:
    PartitionBlockScan m_elementScan;
    BlockScan<glsl::uint> m_blockScan;
    PartitionCombine m_combine;
};

} // namespace wrs
