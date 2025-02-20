#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/wrs/its/ITS.hpp"
#include "src/host/types/glsl.hpp"
#include <stdexcept>
#include <variant>
namespace device {

using WRSConfig = std::variant<ITS::Config>;

[[maybe_unused]]
static std::string wrsConfigName(WRSConfig config) {
    if (std::holds_alternative<ITS::Config>(config)) {
        auto methodConfig = std::get<ITS::Config>(config);
        if (methodConfig.samplingConfig.cooperativeSamplingSize == 0) {
          return fmt::format("ITS-{}", methodConfig.samplingConfig.workgroupSize);
        }else {
          if (methodConfig.samplingConfig.pArraySearch) {
            return fmt::format("ITS-{}-PARRAY-COOP-{}", methodConfig.samplingConfig.workgroupSize, 
                methodConfig.samplingConfig.cooperativeSamplingSize);
          }else {
            return fmt::format("ITS-{}-BINARY-COOP-{}", methodConfig.samplingConfig.workgroupSize, 
                methodConfig.samplingConfig.cooperativeSamplingSize);
          }
        }
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

    std::variant<device::ITS::Buffers> m_internals;

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
    using Method = std::variant<ITS>;

  private:
    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const Config& config) {
        if (std::holds_alternative<ITSConfig>(config)) {
            const auto& methodConfig = std::get<ITSConfig>(config);
            return ITS(context, shaderCompiler, methodConfig);
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
               std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        if (std::holds_alternative<ITS>(m_method)) {
            const ITS& method = std::get<ITS>(m_method);
            method.build(cmd, std::get<ITS::Buffers>(buffers.m_internals), N, profiler);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

    void sample(const merian::CommandBufferHandle& cmd,
                const WRSBuffers& buffers,
                host::glsl::uint N,
                host::glsl::uint S,
                host::glsl::uint seed = 12345u) {
        if (std::holds_alternative<ITS>(m_method)) {
            const ITS& method = std::get<ITS>(m_method);
            method.sample(cmd, std::get<ITS::Buffers>(buffers.m_internals), N, S, seed);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};

} // namespace device
