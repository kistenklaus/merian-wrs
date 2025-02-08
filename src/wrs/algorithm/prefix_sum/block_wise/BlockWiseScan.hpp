#pragma once

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/prefix_sum/block_wise/block_scan/BlockScan.hpp"
#include "src/wrs/algorithm/prefix_sum/block_wise/combine/BlockCombine.hpp"
#include "src/wrs/types/glsl.hpp"
namespace wrs {

struct BlockWiseScanBuffers {
    using Self = BlockWiseScanBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    merian::BufferHandle reductions;
    using ReductionsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ReductionsView = layout::BufferView<ReductionsLayout>;

    merian::BufferHandle prefixSum;
    using PrefixSumLayout = layout::ArrayLayout<float, storageQualifier>;
    using PrefixSumView = layout::BufferView<PrefixSumLayout>;

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
            buffers.reductions = alloc->createBuffer(ReductionsLayout::size(blockCount),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                     memoryMapping);
            buffers.prefixSum = alloc->createBuffer(PrefixSumLayout::size(N),
                                                    vk::BufferUsageFlagBits::eStorageBuffer |
                                                        vk::BufferUsageFlagBits::eTransferSrc,
                                                    memoryMapping);

        } else {
            buffers.elements = alloc->createBuffer(
                ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.reductions =
                alloc->createBuffer(ReductionsLayout::size(blockCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.prefixSum = alloc->createBuffer(
                PrefixSumLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

struct BlockWiseScanConfig {
    const BlockScanConfig elementScanConfig;
    const BlockScanConfig blockScanConfig;
    const BlockCombineConfig blockCombineConfig;

    constexpr BlockWiseScanConfig()
        : elementScanConfig(512,
                            2,
                            BlockScanVariant::SUBGROUP_INTRINSIC |
                                BlockScanVariant::WORKGROUP_SUBGROUP_SCAN,
                            2,
                            true),
          blockScanConfig(512,
                          1,
                          BlockScanVariant::SUBGROUP_INTRINSIC |
                              BlockScanVariant::WORKGROUP_SUBGROUP_SCAN,
                          1,
                          false),
          blockCombineConfig(elementScanConfig) {};

    constexpr explicit BlockWiseScanConfig(BlockScanConfig elementScanConfig,
                                           BlockScanConfig blockScanConfig)
        : elementScanConfig(elementScanConfig), blockScanConfig(blockScanConfig),
          blockCombineConfig(elementScanConfig) {}

    constexpr explicit BlockWiseScanConfig(BlockScanConfig elementScanConfig,
                                           BlockScanConfig blockScanConfig,
                                           BlockCombineConfig combineConfig)
        : elementScanConfig(elementScanConfig), blockScanConfig(blockScanConfig),
          blockCombineConfig(combineConfig) {}

    inline glsl::uint blockSize() const {
        return elementScanConfig.blockSize();
    }

    inline glsl::uint maxElementCount() const {
        return blockScanConfig.blockSize() * elementScanConfig.blockSize();
    }
};

class BlockWiseScan {
  public:
    using Buffers = BlockWiseScanBuffers;

    BlockWiseScan(const merian::ContextHandle& context, BlockWiseScanConfig config = {})
        : m_elementScan(context, config.elementScanConfig),
          m_combine(context, BlockCombineConfig(config.elementScanConfig)),
          m_blockScan(context, config.blockScanConfig) {
        assert(config.elementScanConfig.writeBlockReductions);
    }

    void run(vk::CommandBuffer cmd,
             Buffers buffers,
             glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        assert(N <= m_elementScan.blockSize() * m_blockScan.blockSize());

        // Blockwise scan over elements.
        BlockScanBuffers elementScanBuffers;
        elementScanBuffers.elements = buffers.elements;
        elementScanBuffers.reductions = buffers.reductions;
        elementScanBuffers.prefixSum = buffers.prefixSum;

        if (profiler.has_value()) {
            profiler.value()->start("ElementScan");
            profiler.value()->cmd_start(cmd, "ElementScan");
        }
        m_elementScan.run(cmd, elementScanBuffers, N);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                            buffers.reductions->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                               vk::AccessFlagBits::eShaderRead),
                            {});

        const glsl::uint blockCount = (N + m_elementScan.blockSize() - 1) / m_elementScan.blockSize();
        assert(blockCount <= m_blockScan.blockSize());

        // Inplace scan over block reductions.
        BlockScanBuffers blockScanBuffers;
        blockScanBuffers.elements = buffers.reductions;
        blockScanBuffers.prefixSum = buffers.reductions;
        blockScanBuffers.reductions = nullptr; // disable write to reductions!
                                               //
        if (profiler.has_value()) {
            profiler.value()->start("BlockScan");
            profiler.value()->cmd_start(cmd, "BlockScan");
        }
        m_blockScan.run(cmd, blockScanBuffers, blockCount);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
            {}, {},
            {buffers.reductions->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                vk::AccessFlagBits::eShaderRead),
             buffers.prefixSum->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                               vk::AccessFlagBits::eShaderRead |
                                                   vk::AccessFlagBits::eShaderWrite)},
            {});

        // Combine scan over blocks with scan over elements
        BlockCombineBuffers blockCombineBuffers;
        blockCombineBuffers.blockScan = buffers.reductions;
        blockCombineBuffers.elementScan = buffers.prefixSum;

        if (profiler.has_value()) {
            profiler.value()->start("Combine");
            profiler.value()->cmd_start(cmd, "Combine");
        }
        m_combine.run(cmd, blockCombineBuffers, N);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

  private:
    BlockScan m_elementScan;
    BlockCombine m_combine;
    BlockScan m_blockScan;
};

} // namespace wrs
