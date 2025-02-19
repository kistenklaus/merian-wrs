#pragma once

#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/PrefixSumAllocFlags.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/device/prefix_sum/block_wise/combine/BlockCombine.hpp"
#include "src/host/types/glsl.hpp"
namespace device {

struct BlockWiseScanBuffers {
    using Self = BlockWiseScanBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = host::layout::BufferView<ElementsLayout>;

    merian::BufferHandle reductions;
    using ReductionsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using ReductionsView = host::layout::BufferView<ReductionsLayout>;

    merian::BufferHandle prefixSum;
    using PrefixSumLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using PrefixSumView = host::layout::BufferView<PrefixSumLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N,
                         host::glsl::uint blockCount,
                         PrefixSumAllocFlags allocFlags = PrefixSumAllocFlags::ALLOC_ALL) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            if ((allocFlags & PrefixSumAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       memoryMapping);
            }
            buffers.reductions = alloc->createBuffer(ReductionsLayout::size(blockCount),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                     memoryMapping);
            if ((allocFlags & PrefixSumAllocFlags::ALLOC_PREFIX_SUM) != 0) {
                buffers.prefixSum = alloc->createBuffer(PrefixSumLayout::size(N),
                                                        vk::BufferUsageFlagBits::eStorageBuffer |
                                                            vk::BufferUsageFlagBits::eTransferSrc,
                                                        memoryMapping);
            }

        } else {
            if ((allocFlags & PrefixSumAllocFlags::ALLOC_ELEMENTS) != 0) {
                buffers.elements = alloc->createBuffer(
                    ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }
            buffers.reductions =
                alloc->createBuffer(ReductionsLayout::size(blockCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            if ((allocFlags & PrefixSumAllocFlags::ALLOC_PREFIX_SUM) != 0) {
                buffers.prefixSum = alloc->createBuffer(
                    PrefixSumLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }
        }
        return buffers;
    }
};

struct BlockWiseScanConfig {
    const BlockScanConfig elementScanConfig;
    const BlockScanConfig blockScanConfig;
    const BlockCombineConfig blockCombineConfig;

    constexpr explicit BlockWiseScanConfig(host::glsl::uint workgroupSize,
                                           host::glsl::uint rows,
                                           host::glsl::uint sequentialScanLength,
                                           BlockScanVariant scanVariant)
        : elementScanConfig(workgroupSize, rows, scanVariant, sequentialScanLength, true),
          blockScanConfig(
              512, 8, BlockScanVariant::RANKED_STRIDED | BlockScanVariant::EXCLUSIVE, 1, false),
          blockCombineConfig(workgroupSize, rows * sequentialScanLength, 1, 1) {}

    constexpr BlockWiseScanConfig()
        : elementScanConfig(512, 2, BlockScanVariant::RAKING, 2, true),
          blockScanConfig(512, 1, BlockScanVariant::RAKING, 1, false),
          blockCombineConfig(elementScanConfig) {
        assert(elementScanConfig.blockSize() == blockCombineConfig.blockSize());
        assert((blockScanConfig.variant & BlockScanVariant::EXCLUSIVE) ==
               BlockScanVariant::EXCLUSIVE);
    };

    constexpr explicit BlockWiseScanConfig(BlockScanConfig elementScanConfig,
                                           BlockScanConfig blockScanConfig)
        : elementScanConfig(elementScanConfig), blockScanConfig(blockScanConfig),
          blockCombineConfig(elementScanConfig) {}

    constexpr explicit BlockWiseScanConfig(BlockScanConfig elementScanConfig,
                                           BlockScanConfig blockScanConfig,
                                           BlockCombineConfig combineConfig)
        : elementScanConfig(elementScanConfig), blockScanConfig(blockScanConfig),
          blockCombineConfig(combineConfig) {
        assert(elementScanConfig.blockSize() == blockCombineConfig.blockSize());
        assert((blockScanConfig.variant & BlockScanVariant::EXCLUSIVE) ==
               BlockScanVariant::EXCLUSIVE);
    }

    inline host::glsl::uint blockSize() const {
        return elementScanConfig.blockSize();
    }

    inline host::glsl::uint maxElementCount() const {
        return blockScanConfig.blockSize() * elementScanConfig.blockSize();
    }
};

class BlockWiseScan {
  public:
    using Buffers = BlockWiseScanBuffers;

    using BlockScanKernel = BlockScan<float>;
    using CombineKernel = BlockCombine<float>;

    BlockWiseScan(const merian::ContextHandle& context,
                  const merian::ShaderCompilerHandle& shaderCompiler,
                  BlockWiseScanConfig config = {})
        : m_elementScan(context, shaderCompiler, config.elementScanConfig),
          m_combine(context, shaderCompiler, config.blockCombineConfig),
          m_blockScan(context, shaderCompiler, config.blockScanConfig) {
        assert(config.elementScanConfig.writeBlockReductions);
        assert((config.blockScanConfig.variant & BlockScanVariant::EXCLUSIVE) ==
               BlockScanVariant::EXCLUSIVE);
    }

    void run(const merian::CommandBufferHandle& cmd,
             Buffers buffers,
             host::glsl::uint N,
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

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.reductions->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                        vk::AccessFlagBits::eShaderRead));

        const host::glsl::uint blockCount =
            (N + m_elementScan.blockSize() - 1) / m_elementScan.blockSize();
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

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {buffers.reductions->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                         vk::AccessFlagBits::eShaderRead),
                      buffers.prefixSum->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                        vk::AccessFlagBits::eShaderRead |
                                                            vk::AccessFlagBits::eShaderWrite)});

        // Combine scan over blocks with scan over elements
        CombineKernel::Buffers blockCombineBuffers;
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

    inline host::glsl::uint maxElementCount() const {
        return m_elementScan.blockSize() * m_blockScan.blockSize();
    }

  private:
    BlockScanKernel m_elementScan;
    CombineKernel m_combine;
    BlockScanKernel m_blockScan;
};

} // namespace device
