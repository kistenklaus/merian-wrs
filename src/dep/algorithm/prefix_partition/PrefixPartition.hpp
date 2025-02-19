#pragma once

#include "src/wrs/algorithm/prefix_partition/block_wise/BlockWisePrefixPartition.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartition.hpp"
#include <cassert>
#include <concepts>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <variant>

namespace wrs {

template <typename T>
concept prefix_partition_compatible = std::same_as<float, T>;

using PrefixPartitionConfig =
    std::variant<DecoupledPrefixPartitionConfig, BlockWisePrefixPartitionConfig>;

static std::string prefixPartitionConfigName(PrefixPartitionConfig config) {
    if (std::holds_alternative<DecoupledPrefixPartitionConfig>(config)) {
        auto methodConfig = std::get<DecoupledPrefixPartitionConfig>(config);
        std::string blockScanName;
        if ((methodConfig.blockScanVariant & BlockScanVariant::RAKING) ==
            BlockScanVariant::RAKING) {
            blockScanName = "RAKING";
        } else if ((methodConfig.blockScanVariant & BlockScanVariant::RANKED) ==
                   BlockScanVariant::RANKED) {
            if ((methodConfig.blockScanVariant & BlockScanVariant::STRIDED) ==
                BlockScanVariant::STRIDED) {
                blockScanName = "RANKED-STRIDED";
            } else {
                blockScanName = "RANKED";
            }
        }
        return fmt::format("SingleDispatch-{}-{}-{}", blockScanName, methodConfig.workgroupSize,
                           methodConfig.rows);
    } else if (std::holds_alternative<BlockWisePrefixPartitionConfig>(config)) {
        auto methodConfig = std::get<BlockWisePrefixPartitionConfig>(config);
        std::string blockScanName;
        if ((methodConfig.scanConfig.variant & BlockScanVariant::RAKING) ==
            BlockScanVariant::RAKING) {
            blockScanName = "RAKING";
        } else if ((methodConfig.scanConfig.variant & BlockScanVariant::RANKED) ==
                   BlockScanVariant::RANKED) {
            if ((methodConfig.scanConfig.variant & BlockScanVariant::STRIDED) ==
                BlockScanVariant::STRIDED) {
                blockScanName = "RANKED-STRIDED";
            } else {
                blockScanName = "RANKED";
            }
        }
        return fmt::format("BlockWise-{}-{}-{}", blockScanName,
                           methodConfig.scanConfig.workgroupSize, methodConfig.scanConfig.rows);
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

struct PrefixPartitionBuffers {
    using Self = PrefixPartitionBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    merian::BufferHandle elements;
    template <prefix_partition_compatible T>
    using ElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <prefix_partition_compatible T>
    using ElementsView = layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <prefix_partition_compatible T>
    using PivotLayout = layout::PrimitiveLayout<T, storageQualifier>;
    template <prefix_partition_compatible T> //
    using PivotView = layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <prefix_partition_compatible T>
    using PartitionElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <prefix_partition_compatible T>
    using PartitionElementsView = layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle partitionPrefix;
    template <prefix_partition_compatible T>
    using PartitionPrefixLayout = layout::ArrayLayout<T, storageQualifier>;
    template <prefix_partition_compatible T>
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout<T>>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = layout::PrimitiveLayout<glsl::uint, storageQualifier>;
    using HeavyCountView = layout::BufferView<HeavyCountLayout>;

    struct BlockWiseInternals {
        merian::BufferHandle blockHeavyCount;
        merian::BufferHandle blockHeavyReductions;
        merian::BufferHandle blockLightReductions;
    };

    struct DecoupledInternals {
        merian::BufferHandle decoupledStates;
    };

    std::variant<DecoupledInternals, BlockWiseInternals> m_internalBuffers;

    template <prefix_partition_compatible T>
    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         PrefixPartitionConfig config,
                         glsl::uint N) {

        Self buffers;
        if (std::holds_alternative<DecoupledPrefixPartitionConfig>(config)) {
            const auto methodConfig = std::get<DecoupledPrefixPartitionConfig>(config);
            using MethodBuffers = DecoupledPrefixPartition<T>::Buffers;
            glsl::uint blockCount = (N + methodConfig.blockSize() - 1) / methodConfig.blockSize();
            auto methodBuffers =
                MethodBuffers::template allocate<T>(alloc, memoryMapping, N, blockCount);
            buffers.elements = methodBuffers.elements;
            buffers.pivot = methodBuffers.pivot;
            buffers.partitionIndices = methodBuffers.partitionIndices;
            buffers.partitionElements = methodBuffers.partitionElements;
            buffers.partitionPrefix = methodBuffers.partitionPrefix;
            buffers.heavyCount = methodBuffers.heavyCount;
            DecoupledInternals internals;
            internals.decoupledStates = methodBuffers.decoupledStates;
            buffers.m_internalBuffers = internals;

        } else if (std::holds_alternative<BlockWisePrefixPartitionConfig>(config)) {
            const auto methodConfig = std::get<BlockWisePrefixPartitionConfig>(config);
            using MethodBuffers = BlockWisePrefixPartition<T>::Buffers;
            const glsl::uint blockCount =
                (N + methodConfig.blockSize() - 1) / methodConfig.blockSize();
            auto methodBuffers =
                MethodBuffers::template allocate<T>(alloc, memoryMapping, N, blockCount);
            buffers.elements = methodBuffers.elements;
            buffers.pivot = methodBuffers.pivot;
            buffers.partitionIndices = methodBuffers.partitionIndices;
            buffers.partitionElements = methodBuffers.partitionElements;
            buffers.partitionPrefix = methodBuffers.partitionPrefix;
            buffers.heavyCount = methodBuffers.heavyCount;
            BlockWiseInternals internals;
            internals.blockHeavyCount = methodBuffers.blockHeavyCount;
            internals.blockHeavyReductions = methodBuffers.blockHeavyReductions;
            internals.blockLightReductions = methodBuffers.blockLightReductions;
            buffers.m_internalBuffers = internals;
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
        return buffers;
    }
};

template <prefix_partition_compatible T> class PrefixPartition {
  public:
    using Buffers = PrefixPartitionBuffers;
    using Config = PrefixPartitionConfig;

  private:
    using Method = std::variant<DecoupledPrefixPartition<T>, BlockWisePrefixPartition<T>>;
    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const PrefixPartitionConfig& config) {
        if (std::holds_alternative<DecoupledPrefixPartitionConfig>(config)) {
            return DecoupledPrefixPartition<T>(context, shaderCompiler,
                                               std::get<DecoupledPrefixPartitionConfig>(config));
        } else if (std::holds_alternative<BlockWisePrefixPartitionConfig>(config)) {
            return BlockWisePrefixPartition<T>(context, shaderCompiler,
                                               std::get<BlockWisePrefixPartitionConfig>(config));
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    PrefixPartition(const merian::ContextHandle& context,
                    const merian::ShaderCompilerHandle& shaderCompiler,
                    const PrefixPartitionConfig config)
        : m_method(createMethod(context, shaderCompiler, config)) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        if (std::holds_alternative<DecoupledPrefixPartition<T>>(m_method)) {
            assert(std::holds_alternative<Buffers::DecoupledInternals>(buffers.m_internalBuffers));
            using MethodBuffers = DecoupledPrefixPartition<T>::Buffers;
            MethodBuffers methodBuffers;
            methodBuffers.elements = buffers.elements;
            methodBuffers.pivot = buffers.pivot;
            methodBuffers.partitionIndices = buffers.partitionIndices;
            methodBuffers.partitionElements = buffers.partitionElements;
            methodBuffers.partitionPrefix = buffers.partitionPrefix;
            methodBuffers.heavyCount = buffers.heavyCount;
            methodBuffers.decoupledStates =
                std::get<Buffers::DecoupledInternals>(buffers.m_internalBuffers).decoupledStates;

            std::get<DecoupledPrefixPartition<T>>(m_method).run(cmd, methodBuffers, N);
        } else if (std::holds_alternative<BlockWisePrefixPartition<T>>(m_method)) {
            assert(std::holds_alternative<Buffers::BlockWiseInternals>(buffers.m_internalBuffers));
            using MethodBuffers = BlockWisePrefixPartition<T>::Buffers;
            MethodBuffers methodBuffers;
            methodBuffers.elements = buffers.elements;
            methodBuffers.pivot = buffers.pivot;
            methodBuffers.partitionIndices = buffers.partitionIndices;
            methodBuffers.partitionElements = buffers.partitionElements;
            methodBuffers.partitionPrefix = buffers.partitionPrefix;
            methodBuffers.heavyCount = buffers.heavyCount;
            methodBuffers.blockHeavyCount =
                std::get<Buffers::BlockWiseInternals>(buffers.m_internalBuffers).blockHeavyCount;
            methodBuffers.blockHeavyReductions =
                std::get<Buffers::BlockWiseInternals>(buffers.m_internalBuffers)
                    .blockHeavyReductions;
            methodBuffers.blockLightReductions =
                std::get<Buffers::BlockWiseInternals>(buffers.m_internalBuffers)
                    .blockLightReductions;

            std::get<BlockWisePrefixPartition<T>>(m_method).run(cmd, methodBuffers, N, profiler);
        }
    }

  private:
    Method m_method;
};

} // namespace wrs
