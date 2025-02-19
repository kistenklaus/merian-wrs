#pragma once

#include "src/device/partition/PartitionAllocFlags.hpp"
#include "src/device/partition/block_wise/BlockWisePartition.hpp"
#include "src/device/partition/decoupled/DecoupledPartition.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/layout/PrimitiveLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <concepts>
#include <fmt/format.h>
#include <stdexcept>
#include <variant>

namespace device {

typedef std::variant<DecoupledPartitionConfig, BlockWisePartitionConfig> PartitionConfig;

template <typename T>
concept partition_compatible_type = std::same_as<host::glsl::f32, T>;

[[maybe_unused]]
static std::string partitionConfigName(PartitionConfig config) {
    if (std::holds_alternative<DecoupledPartitionConfig>(config)) {
        DecoupledPartitionConfig instanceConfig = std::get<DecoupledPartitionConfig>(config);
        std::string blockScanName = blockScanVariantName(instanceConfig.blockScanVariant);

        return fmt::format("SingleDispatch-{}-{}-{}", blockScanName, instanceConfig.workgroupSize,
                           instanceConfig.rows);
    } else if (std::holds_alternative<BlockWisePartitionConfig>(config)) {
        BlockWisePartitionConfig instanceConfig = std::get<BlockWisePartitionConfig>(config);
        std::string blockScanName = blockScanVariantName(instanceConfig.elementScanConfig.variant);
        return fmt::format("BlockWise-{}-{}-{}-{}", blockScanName,
                           instanceConfig.elementScanConfig.workgroupSize,
                           instanceConfig.elementScanConfig.rows,
                           instanceConfig.elementScanConfig.sequentialScanLength);
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

class PartitionBuffers {
  public:
    using Self = PartitionBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    merian::BufferHandle elements;
    template <partition_compatible_type T>
    using ElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <partition_compatible_type T>
    using ElementsView = host::layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle pivot;
    template <partition_compatible_type T>
    using PivotLayout = host::layout::PrimitiveLayout<T, storageQualifier>;
    template <partition_compatible_type T> //
    using PivotView = host::layout::BufferView<PivotLayout<T>>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionElements;
    template <partition_compatible_type T>
    using PartitionElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <partition_compatible_type T>
    using PartitionElementsView = host::layout::BufferView<PartitionElementsLayout<T>>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    struct DecoupledInternal {
        merian::BufferHandle decoupledStates;
    };

    struct BlockWiseInternal {
        merian::BufferHandle indicies;
        merian::BufferHandle blockIndicies;
    };

    std::variant<DecoupledInternal, BlockWiseInternal> m_internalBuffers;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         PartitionConfig config,
                         host::glsl::uint N,
                         PartitionAllocFlags allocFlags = PartitionAllocFlags::ALLOC_ALL) {

        Self buffers;
        if (std::holds_alternative<DecoupledPartitionConfig>(config)) {
            auto instanceConfig = std::get<DecoupledPartitionConfig>(config);
            host::glsl::uint blockCount =
                (N + instanceConfig.blockSize() - 1) / instanceConfig.blockSize();
            using InstanceBuffers = DecoupledPartition::Buffers;
            InstanceBuffers instanceBuffers =
                InstanceBuffers::allocate(alloc, memoryMapping, N, blockCount, allocFlags);
            assert(instanceBuffers.elements != nullptr);
            buffers.elements = instanceBuffers.elements;
            buffers.pivot = instanceBuffers.pivot;
            buffers.partitionIndices = instanceBuffers.partitionIndices;
            buffers.partitionElements = instanceBuffers.partition;
            buffers.heavyCount = instanceBuffers.heavyCount;
            DecoupledInternal internalBuffers;
            internalBuffers.decoupledStates = instanceBuffers.decoupledStates;
            buffers.m_internalBuffers = internalBuffers;

        } else if (std::holds_alternative<BlockWisePartitionConfig>(config)) {
            auto instanceConfig = std::get<BlockWisePartitionConfig>(config);
            host::glsl::uint blockCount =
                (N + instanceConfig.blockSize() - 1) / instanceConfig.blockSize();
            using InstanceBuffers = BlockWisePartition::Buffers;
            InstanceBuffers instanceBuffers =
                InstanceBuffers::allocate(alloc, memoryMapping, N, blockCount, allocFlags);
            buffers.elements = instanceBuffers.elements;
            buffers.pivot = instanceBuffers.pivot;
            buffers.partitionIndices = instanceBuffers.partitionIndices;
            buffers.partitionElements = instanceBuffers.partition;
            buffers.heavyCount = instanceBuffers.heavyCount;
            BlockWiseInternal internalBuffers;
            internalBuffers.indicies = instanceBuffers.indices;
            internalBuffers.blockIndicies = instanceBuffers.blockIndices;
            buffers.m_internalBuffers = internalBuffers;
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED-YET");
        }
        return buffers;
    }
};

template <typename T> class Partition {
    using Method = std::variant<DecoupledPartition, BlockWisePartition>;

    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               PartitionConfig config) {
        if (std::holds_alternative<DecoupledPartitionConfig>(config)) {
            return DecoupledPartition(context, shaderCompiler,
                                      std::get<DecoupledPartitionConfig>(config));
        } else if (std::holds_alternative<BlockWisePartitionConfig>(config)) {
            return BlockWisePartition(context, shaderCompiler,
                                      std::get<BlockWisePartitionConfig>(config));
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    using Buffers = PartitionBuffers;
    using Config = PartitionConfig;

    Partition(const merian::ContextHandle& context,
              const merian::ShaderCompilerHandle& shaderCompiler,
              PartitionConfig config)
        : m_method(createMethod(context, shaderCompiler, config)) {}

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N) {
        if (std::holds_alternative<DecoupledPartition>(m_method)) {
            using InstanceBuffers = DecoupledPartition::Buffers;
            InstanceBuffers instanceBuffers;
            instanceBuffers.elements = buffers.elements;
            instanceBuffers.pivot = buffers.pivot;
            assert(std::holds_alternative<Buffers::DecoupledInternal>(buffers.m_internalBuffers));
            instanceBuffers.decoupledStates =
                std::get<Buffers::DecoupledInternal>(buffers.m_internalBuffers).decoupledStates;
            instanceBuffers.partitionIndices = buffers.partitionIndices;
            instanceBuffers.partition = buffers.partitionElements;
            instanceBuffers.heavyCount = buffers.heavyCount;

            std::get<DecoupledPartition>(m_method).run(cmd, instanceBuffers, N);
        } else if (std::holds_alternative<BlockWisePartition>(m_method)) {
            using InstanceBuffers = BlockWisePartition::Buffers;
            InstanceBuffers instanceBuffers;
            instanceBuffers.elements = buffers.elements;
            instanceBuffers.pivot = buffers.pivot;
            instanceBuffers.partitionIndices = buffers.partitionIndices;
            assert(std::holds_alternative<Buffers::BlockWiseInternal>(buffers.m_internalBuffers));
            instanceBuffers.indices =
                std::get<Buffers::BlockWiseInternal>(buffers.m_internalBuffers).indicies;
            instanceBuffers.blockIndices =
                std::get<Buffers::BlockWiseInternal>(buffers.m_internalBuffers).blockIndicies;

            instanceBuffers.partition = buffers.partitionElements;
            instanceBuffers.heavyCount = buffers.heavyCount;

            std::get<BlockWisePartition>(m_method).run(cmd, instanceBuffers, N);

        } else {
            throw std::runtime_error("Unsupported method");
        }
    }

    inline host::glsl::uint maxElementCount() {
        if (std::holds_alternative<DecoupledPartition>(m_method)) {
            return (1 << 28);
        } else if (std::holds_alternative<BlockWisePartition>(m_method)) {
            return std::get<BlockWisePartition>(m_method).maxElementCount();
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

    std::variant<DecoupledPartition, BlockWisePartition> m_method;
};

} // namespace device
