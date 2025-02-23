#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/wrs/alias/AliasTable.hpp"
#include "src/device/wrs/its/ITS.hpp"
#include "src/host/types/glsl.hpp"
#include <stdexcept>
#include <variant>
namespace device {

using WRSConfig = std::variant<ITS::Config, AliasTable::Config>;

[[maybe_unused]]
static std::string wrsConfigName(WRSConfig config) {
    if (std::holds_alternative<ITS::Config>(config)) {
        auto methodConfig = std::get<ITS::Config>(config);
        if (methodConfig.samplingConfig.cooperativeSamplingSize == 0) {
            return fmt::format("ITS-{}", methodConfig.samplingConfig.workgroupSize);
        } else {
            if (methodConfig.samplingConfig.pArraySearch) {
                return fmt::format("ITS-{}-PARRAY-COOP-{}",
                                   methodConfig.samplingConfig.workgroupSize,
                                   methodConfig.samplingConfig.cooperativeSamplingSize);
            } else {
                return fmt::format("ITS-{}-BINARY-COOP-{}",
                                   methodConfig.samplingConfig.workgroupSize,
                                   methodConfig.samplingConfig.cooperativeSamplingSize);
            }
        }
    } else if (std::holds_alternative<AliasTable::Config>(config)) {
        auto methodConfig = std::get<AliasTable::Config>(config);
        return methodConfig.name();
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

struct WRSBuffers {
  public:
    using Self = WRSBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using SamplesView = host::layout::BufferView<SamplesLayout>;

    std::variant<device::ITS::Buffers, device::AliasTable::Buffers> m_internals;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t S,
                         WRSConfig config) {
        Self buffers;
        if (std::holds_alternative<ITS::Config>(config)) {
            ITS::Buffers methodBuffers = ITS::Buffers::allocate(
                alloc, memoryMapping, N, S, std::get<ITS::Config>(config).prefixSumConfig);
            buffers.weights = methodBuffers.weights;
            buffers.samples = methodBuffers.samples;
            buffers.m_internals = methodBuffers;
        } else if (std::holds_alternative<AliasTable::Config>(config)) {
            AliasTable::Buffers methodBuffers = AliasTable::Buffers::allocate(
                alloc, memoryMapping, std::get<AliasTable::Config>(config), N, S);
            buffers.weights = methodBuffers.weights;
            buffers.samples = methodBuffers.samples;
            buffers.m_internals = methodBuffers;
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
        return buffers;
    }
};

class WRS {
  public:
    using Buffers = WRSBuffers;
    using Config = WRSConfig;
    using Method = std::variant<ITS, AliasTable>;

  private:
    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const Config& config) {
        if (std::holds_alternative<ITSConfig>(config)) {
            const auto& methodConfig = std::get<ITSConfig>(config);
            return ITS(context, shaderCompiler, methodConfig);
        } else if (std::holds_alternative<AliasTable::Config>(config)) {
            const auto& methodConfig = std::get<AliasTable::Config>(config);
            return AliasTable(context, shaderCompiler, methodConfig);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    explicit WRS(const merian::ContextHandle& context,
                 const merian::ShaderCompilerHandle& shaderCompiler,
                 Config config)
        : m_method(createMethod(context, shaderCompiler, config)) {}

    void build(const merian::CommandBufferHandle& cmd,
               const WRSBuffers& buffers,
               host::glsl::uint N,
               std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        if (std::holds_alternative<ITS>(m_method)) {
            const ITS& method = std::get<ITS>(m_method);
            ITS::Buffers itsBuffers = std::get<ITS::Buffers>(buffers.m_internals);
            itsBuffers.weights = buffers.weights;
            method.build(cmd, itsBuffers, N, profiler);
        } else if (std::holds_alternative<AliasTable>(m_method)) {
            const auto& alias = std::get<AliasTable>(m_method);
            AliasTable::Buffers aliasBuffers = std::get<AliasTable::Buffers>(buffers.m_internals);
            aliasBuffers.weights = buffers.weights;
            alias.build(cmd, aliasBuffers, N, profiler);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

    void sample(const merian::CommandBufferHandle& cmd,
                const WRSBuffers& buffers,
                host::glsl::uint N,
                host::glsl::uint S,
                host::glsl::uint seed = 12345u) const {
        if (std::holds_alternative<ITS>(m_method)) {
            const ITS& method = std::get<ITS>(m_method);
            ITS::Buffers itsBuffers = std::get<ITS::Buffers>(buffers.m_internals);
            itsBuffers.samples = buffers.samples;
            method.sample(cmd, itsBuffers, N, S, seed);
        } else if (std::holds_alternative<AliasTable>(m_method)) {
            const AliasTable& method = std::get<AliasTable>(m_method);
            AliasTable::Buffers aliasBuffers = std::get<AliasTable::Buffers>(buffers.m_internals);
            aliasBuffers.samples = buffers.samples;
            method.sample(cmd, aliasBuffers, N, S, seed);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};

} // namespace device
