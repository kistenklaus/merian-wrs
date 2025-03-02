#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include <stdexcept>
#include <variant>
namespace device {

using PRNGConfig = std::variant<PhiloxConfig>;

struct PRNGBuffers {
    merian::BufferHandle samples;
};

class PRNG {
  public:
    using Buffers = PRNGBuffers;
    using Config = PRNGConfig;

  private:
    using Method = std::variant<Philox>;
    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const Config& config) {
        if (std::holds_alternative<PhiloxConfig>(config)) {
            auto methodConfig = std::get<PhiloxConfig>(config);
            return Philox(context, shaderCompiler, methodConfig);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    PRNG(const merian::ContextHandle& context,
         const merian::ShaderCompilerHandle& shaderCompiler,
         const Config& config)
        : m_method(createMethod(context, shaderCompiler, config)) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint S,
             host::glsl::uint seed = 12345u) {
        if (std::holds_alternative<Philox>(m_method)) {
            const auto& method = std::get<Philox>(m_method);
            Philox::Buffers methodBuffers;
            methodBuffers.samples = buffers.samples;
            method.run(cmd, methodBuffers, S, seed);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};
} // namespace device
