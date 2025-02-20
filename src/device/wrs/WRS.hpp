#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/wrs/its/ITS.hpp"
#include "src/host/types/glsl.hpp"
#include <stdexcept>
#include <variant>
namespace device {

using WRSConfig = std::variant<device::ITS::Config>;

struct WRSBuffers {
  public:
    using Self = WRSBuffers;
    merian::BufferHandle weights;

    merian::BufferHandle samples;

    using Internals = std::variant<device::ITS::Buffers>;

    Internals m_internals;
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
                host::glsl::uint S) {
        if (std::holds_alternative<ITS>(m_method)) {
            const ITS& method = std::get<ITS>(m_method);
            method.sample(cmd, std::get<ITS::Buffers>(buffers.m_internals), N, S);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};

} // namespace device
