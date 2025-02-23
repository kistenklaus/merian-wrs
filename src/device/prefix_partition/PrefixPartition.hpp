#pragma once

#include "src/device/prefix_partition/PrefixPartitionAllocFlags.hpp"
#include "src/device/prefix_partition/block_wise/BlockWisePrefixPartition.hpp"
#include "src/device/prefix_partition/decoupled/DecoupledPrefixPartition.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include <cassert>
#include <concepts>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <variant>

namespace device {

template <typename T>
concept prefix_partition_compatible = std::same_as<float, T>;

using PrefixPartitionConfig =
    std::variant<DecoupledPrefixPartitionConfig, BlockWisePrefixPartitionConfig>;

[[maybe_unused]]
static std::string prefixPartitionConfigName(const PrefixPartitionConfig& config) {
    if (std::holds_alternative<DecoupledPrefixPartitionConfig>(config)) {
        const auto& methodConfig = std::get<DecoupledPrefixPartitionConfig>(config);
        std::string blockScanName = blockScanVariantName(methodConfig.blockScanVariant);
        return fmt::format("SingleDispatch-{}-{}-{}", blockScanName, methodConfig.workgroupSize,
                           methodConfig.rows);
    } else if (std::holds_alternative<BlockWisePrefixPartitionConfig>(config)) {
        const auto& methodConfig = std::get<BlockWisePrefixPartitionConfig>(config);
        std::string blockScanName = blockScanVariantName(methodConfig.scanConfig.variant);
        return fmt::format("BlockWise-{}-{}-{}", blockScanName,
                           methodConfig.scanConfig.workgroupSize, methodConfig.scanConfig.rows);
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

struct PrefixPartitionBuffers {
    using Self = PrefixPartitionBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    merian::BufferHandle elements;
    template <prefix_partition_compatible T>
    using ElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <prefix_partition_compatible T>
    using ElementsView = host::layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <prefix_partition_compatible T>
    using PivotLayout = host::layout::PrimitiveLayout<T, storageQualifier>;
    template <prefix_partition_compatible T> //
    using PivotView = host::layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <prefix_partition_compatible T>
    using PartitionElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <prefix_partition_compatible T>
    using PartitionElementsView = host::layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle partitionPrefix;
    template <prefix_partition_compatible T>
    using PartitionPrefixLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <prefix_partition_compatible T>
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout<T>>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    struct BlockWiseInternals {
        merian::BufferHandle blockHeavyCount;
        merian::BufferHandle blockHeavyReductions;
        merian::BufferHandle blockLightReductions;
    };

    struct DecoupledInternals {
        merian::BufferHandle decoupledStates;
    };
    using Internals = std::variant<DecoupledInternals, BlockWiseInternals>;
    Internals m_internalBuffers;

    template <prefix_partition_compatible T>
    static Self
    allocate(const merian::ResourceAllocatorHandle& alloc,
             merian::MemoryMappingType memoryMapping,
             PrefixPartitionConfig config,
             host::glsl::uint N,
             PrefixPartitionAllocFlags allocFlags = PrefixPartitionAllocFlags::ALLOC_ALL) {

        Self buffers;
        if (std::holds_alternative<DecoupledPrefixPartitionConfig>(config)) {
            const auto methodConfig = std::get<DecoupledPrefixPartitionConfig>(config);
            using MethodBuffers = DecoupledPrefixPartition<T>::Buffers;
            host::glsl::uint blockCount =
                (N + methodConfig.blockSize() - 1) / methodConfig.blockSize();
            auto methodBuffers =
                MethodBuffers::template allocate<T>(alloc, memoryMapping, N, blockCount, allocFlags);
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
            const host::glsl::uint blockCount =
                (N + methodConfig.blockSize() - 1) / methodConfig.blockSize();
            auto methodBuffers =
                MethodBuffers::template allocate<T>(alloc, memoryMapping, N, blockCount, allocFlags);
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
                               const PrefixPartitionConfig& config,
                               bool writePartitionElements) {
        if (std::holds_alternative<DecoupledPrefixPartitionConfig>(config)) {
            return DecoupledPrefixPartition<T>(context, shaderCompiler,
                                               std::get<DecoupledPrefixPartitionConfig>(config),
                                               writePartitionElements);
        } else if (std::holds_alternative<BlockWisePrefixPartitionConfig>(config)) {
            assert(writePartitionElements); // NOT implemented yet!
            return BlockWisePrefixPartition<T>(context, shaderCompiler,
                                               std::get<BlockWisePrefixPartitionConfig>(config));
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    PrefixPartition(const merian::ContextHandle& context,
                    const merian::ShaderCompilerHandle& shaderCompiler,
                    const PrefixPartitionConfig config,
                    bool writePartitionElements)
        : m_method(createMethod(context, shaderCompiler, config, writePartitionElements)) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
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

            std::get<DecoupledPrefixPartition<T>>(m_method).run(cmd, methodBuffers, N, profiler);
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

} // namespace device
