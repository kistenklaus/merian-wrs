#pragma once

#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/device/wrs/alias/psa/layout/split.hpp"
#include "src/device/wrs/alias/psa/split/Split.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPackAllocFlags.hpp"
#include "src/device/wrs/alias/psa/splitpack/inline/InlineSplitPack.hpp"
#include "src/device/wrs/alias/psa/splitpack/serial/SerialSplitPack.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include "vulkan/vulkan_core.h"
#include "vulkan/vulkan_enums.hpp"
#include <fmt/format.h>
#include <stdexcept>
#include <variant>

namespace device {

using SplitPackConfig = std::variant<InlineSplitPack::Config, SerialSplitPack::Config>;

constexpr host::glsl::uint splitPackConfigSplitSize(const SplitPackConfig& config) {
    if (std::holds_alternative<InlineSplitPack::Config>(config)) {
        const auto& methodConfig = std::get<InlineSplitPack::Config>(config);
        return methodConfig.splitSize;
    } else if (std::holds_alternative<SerialSplitPack::Config>(config)) {
        const auto& methodConfig = std::get<SerialSplitPack::Config>(config);
        return splitConfigSplitSize(methodConfig.splitConfig);
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

[[maybe_unused]]
static std::string splitPackConfigName(const SplitPackConfig& config) {
    if (std::holds_alternative<InlineSplitPack::Config>(config)) {
        const auto& methodConfig = std::get<InlineSplitPack::Config>(config);
        return fmt::format("InlineSplitPack-{}", methodConfig.splitSize);
    } else if (std::holds_alternative<SerialSplitPack::Config>(config)) {
        const auto& methodConfig = std::get<SerialSplitPack::Config>(config);
        return fmt::format("Serial-{}-{}", splitConfigName(methodConfig.splitConfig),
            packConfigName(methodConfig.packConfig));
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

struct SplitPackBuffers {
    using Self = SplitPackBuffers;
    using weight_type = host::glsl::f32;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    merian::BufferHandle mean;
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableLayout = device::details::AliasTableLayout;
    using AliasTableView = host::layout::BufferView<AliasTableLayout>;

    merian::BufferHandle partitionElements; // optional
    using PartitionElementsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using PartitionElementsView = host::layout::BufferView<PartitionElementsLayout>;

    struct InlineInternals {};

    struct SerialInternals {
        merian::BufferHandle splits;
    };

    using Internals = std::variant<InlineInternals, SerialInternals>;
    Internals m_internals;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         const SplitPackConfig config,
                         host::glsl::uint N,
                         SplitPackAllocFlags allocFlags = SplitPackAllocFlags::ALLOC_ALL);
};

class SplitPack {
  public:
    using Buffers = SplitPackBuffers;
    using Config = SplitPackConfig;

  private:
    using Method = std::variant<InlineSplitPack, SerialSplitPack>;

    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const Config& config,
                               bool usePartitionElements) {
        if (std::holds_alternative<InlineSplitPack::Config>(config)) {
            const auto& methodConfig = std::get<InlineSplitPack::Config>(config);
            return InlineSplitPack(context, shaderCompiler, methodConfig, usePartitionElements);
        } else if (std::holds_alternative<SerialSplitPack::Config>(config)) {
            const auto& methodConfig = std::get<SerialSplitPack::Config>(config);
            return SerialSplitPack(context, shaderCompiler, methodConfig, usePartitionElements);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    explicit SplitPack(const merian::ContextHandle& context,
                       const merian::ShaderCompilerHandle& shaderCompiler,
                       const Config& config,
                       bool usePartitionElements)
        : m_method(createMethod(context, shaderCompiler, config, usePartitionElements)) {}

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N,
        std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        if (std::holds_alternative<InlineSplitPack>(m_method)) {
            assert(std::holds_alternative<Buffers::InlineInternals>(buffers.m_internals));
            InlineSplitPack::Buffers methodBuffers;
            methodBuffers.aliasTable = buffers.aliasTable;
            methodBuffers.heavyCount = buffers.heavyCount;
            methodBuffers.mean = buffers.mean;
            methodBuffers.partitionElements = buffers.partitionElements;
            methodBuffers.partitionIndices = buffers.partitionIndices;
            methodBuffers.partitionPrefix = buffers.partitionPrefix;
            methodBuffers.weights = buffers.weights;

            std::get<InlineSplitPack>(m_method).run(cmd, methodBuffers, N, profiler);
        } else if (std::holds_alternative<SerialSplitPack>(m_method)) {
            assert(std::holds_alternative<Buffers::SerialInternals>(buffers.m_internals));

            SerialSplitPack::Buffers methodBuffers;
            methodBuffers.weights = buffers.weights;
            methodBuffers.partitionPrefix = buffers.partitionPrefix;
            methodBuffers.partitionIndices = buffers.partitionIndices;
            methodBuffers.partitionElements = buffers.partitionElements;
            methodBuffers.heavyCount = buffers.heavyCount;
            methodBuffers.aliasTable = buffers.aliasTable;
            methodBuffers.mean = buffers.mean;
            methodBuffers.splits = std::get<Buffers::SerialInternals>(buffers.m_internals).splits;

            std::get<SerialSplitPack>(m_method).run(cmd, methodBuffers, N, profiler);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};

} // namespace device
