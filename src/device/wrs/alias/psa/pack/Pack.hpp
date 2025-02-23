#pragma once

#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/device/wrs/alias/psa/layout/split.hpp"
#include "src/device/wrs/alias/psa/pack/PackAllocFlags.hpp"
#include "src/device/wrs/alias/psa/pack/scalar/ScalarPack.hpp"
#include "src/device/wrs/alias/psa/pack/subgroup/SubgroupPack.hpp"

#include <cassert>
#include <fmt/format.h>
#include <stdexcept>
#include <variant>

namespace device {

using PackConfig = std::variant<ScalarPack::Config, SubgroupPack::Config>;

constexpr host::glsl::uint packConfigSplitSize(const PackConfig& config) {
    if (std::holds_alternative<ScalarPack::Config>(config)) {
        return std::get<ScalarPack::Config>(config).splitSize;
    } else if (std::holds_alternative<SubgroupPack::Config>(config)) {
        return std::get<SubgroupPack::Config>(config).splitSize;
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

constexpr std::string packConfigName(const PackConfig& config) {
    if (std::holds_alternative<ScalarPack::Config>(config)) {
        /* const auto& methodConfig = std::get<ScalarPack::Config>(config); */
        return "ScalarPack";
    } else if (std::holds_alternative<SubgroupPack::Config>(config)) {
        const auto& methodConfig = std::get<SubgroupPack::Config>(config);
        return fmt::format("SubgroupPack-32/{}", methodConfig.subgroupSplit);
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

struct PackBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using weight_type = host::glsl::f32;
    using Self = PackBuffers;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean;
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using PartitionIndicesView = host::layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle splits;
    using SplitsLayout = device::details::SplitsLayout;
    using SplitsView = host::layout::BufferView<SplitsLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableLayout = device::details::AliasTableLayout;
    using AliasTableView = host::layout::BufferView<AliasTableLayout>;

    // optional buffers
    merian::BufferHandle partitionElements;
    using PartitionElementsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using PartitionElementsView = host::layout::BufferView<PartitionIndicesLayout>;

    static Self allocate(merian::ResourceAllocatorHandle alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t K,
                         PackAllocFlags allocFlags = PackAllocFlags::ALLOC_DEFAULT);
};

class Pack {
  public:
    using Buffers = PackBuffers;
    using Config = PackConfig;

  private:
    using Method = std::variant<ScalarPack, SubgroupPack>;

    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const Config& config,
                               const bool usePartitionElements) {
        if (std::holds_alternative<ScalarPack::Config>(config)) {
            const auto& methodConfig = std::get<ScalarPack::Config>(config);
            return ScalarPack(context, shaderCompiler, methodConfig, usePartitionElements);
        } else if (std::holds_alternative<SubgroupPack::Config>(config)) {
            const auto& methodConfig = std::get<SubgroupPack::Config>(config);
            return SubgroupPack(context, shaderCompiler, methodConfig, usePartitionElements);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    Pack(const merian::ContextHandle& context,
         const merian::ShaderCompilerHandle& shaderCompiler,
         const Config config,
         bool usePartitionElements)
        : m_method(createMethod(context, shaderCompiler, config, usePartitionElements)) {}

    void
    run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N,
        std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {

        if (std::holds_alternative<ScalarPack>(m_method)) {
            const auto& method = std::get<ScalarPack>(m_method);
            ScalarPack::Buffers methodBuffers;
            methodBuffers.weights = buffers.weights;
            methodBuffers.mean = buffers.mean;
            methodBuffers.heavyCount = buffers.heavyCount;
            methodBuffers.partitionIndices = buffers.partitionIndices;
            methodBuffers.partitionElements = buffers.partitionElements;
            methodBuffers.splits = buffers.splits;
            methodBuffers.aliasTable = buffers.aliasTable;
            method.run(cmd, methodBuffers, N, profiler);
        } else if (std::holds_alternative<SubgroupPack>(m_method)) {
            const auto& method = std::get<SubgroupPack>(m_method);
            SubgroupPack::Buffers methodBuffers;
            methodBuffers.weights = buffers.weights;
            methodBuffers.mean = buffers.mean;
            methodBuffers.heavyCount = buffers.heavyCount;
            methodBuffers.partitionIndices = buffers.partitionIndices;
            methodBuffers.partition = buffers.partitionElements;
            methodBuffers.splits = buffers.splits;
            methodBuffers.aliasTable = buffers.aliasTable;
            method.run(cmd, methodBuffers, N, profiler);

        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

    inline host::glsl::uint splitSize() const {
        if (std::holds_alternative<ScalarPack>(m_method)) {
            const auto& method = std::get<ScalarPack>(m_method);
            return method.splitSize();
        } else if (std::holds_alternative<SubgroupPack>(m_method)) {
            const auto& method = std::get<SubgroupPack>(m_method);
            return method.splitSize();
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};

} // namespace device
  //
