#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/algorithm/mean/MeanAllocFlags.hpp"
#include "src/wrs/algorithm/mean/atomic/AtomicMean.hpp"
#include "src/wrs/algorithm/mean/decoupled/DecoupledMean.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include <cassert>
#include <concepts>
#include <stdexcept>
#include <variant>
namespace wrs {

using MeanConfig = std::variant<AtomicMeanConfig, DecoupledMeanConfig>;

[[maybe_unused]]
static std::string meanConfigName(const MeanConfig& config) {
    if (std::holds_alternative<AtomicMeanConfig>(config)) {
        const auto& methodConfig = std::get<AtomicMeanConfig>(config);
        return fmt::format("Atomic-{}-{}", methodConfig.workgroupSize, methodConfig.rows);
    } else if (std::holds_alternative<DecoupledMeanConfig>(config)) {
        const auto& methodConfig = std::get<DecoupledMeanConfig>(config);
        return fmt::format("Decoupled-{}-{}", methodConfig.workgroupSize, methodConfig.rows);
    } else {
        throw std::runtime_error("NOT-IMPLEMENTED");
    }
}

template <typename T>
concept mean_compatible = std::same_as<float, T>;

class MeanBuffers {
  public:
    using Self = MeanBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    template <mean_compatible T> using ElementsLayout = layout::ArrayLayout<T, storageQualifier>;
    template <mean_compatible T> using ElementsView = layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle mean;
    template <mean_compatible T> using MeanLayout = layout::PrimitiveLayout<T, storageQualifier>;
    template <mean_compatible T> using MeanView = layout::BufferView<MeanLayout<T>>;

    struct DecoupledInternals {
        merian::BufferHandle decoupledStates;
    };

    struct AtomicInternals {};

    std::variant<DecoupledInternals, AtomicInternals> m_internalBuffers;

    template <mean_compatible T>
    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         MeanConfig config,
                         glsl::uint N,
                         MeanAllocFlags allocFlags = MeanAllocFlags::ALLOC_ALL) {
        Self buffers;
        if (std::holds_alternative<AtomicMeanConfig>(config)) {
            /* const auto methodConfig = std::get<AtomicMeanConfig>(config); */
            using MethodBuffers = AtomicMean::Buffers;
            auto methodBuffers = MethodBuffers::allocate(alloc, memoryMapping, N, allocFlags);
            buffers.elements = methodBuffers.elements;
            buffers.mean = methodBuffers.mean;
            AtomicInternals internals;
            buffers.m_internalBuffers = internals; // taged union!!
        } else if (std::holds_alternative<DecoupledMeanConfig>(config)) {
            const auto methodConfig = std::get<DecoupledMeanConfig>(config);
            using MethodBuffers = DecoupledMean::Buffers;
            glsl::uint blockCount = (N + methodConfig.blockSize() - 1) / methodConfig.blockSize();
            auto methodBuffers = MethodBuffers::allocate(alloc, N, blockCount, memoryMapping, allocFlags);
            buffers.elements = methodBuffers.elements;
            buffers.mean = methodBuffers.mean;
            DecoupledInternals internals;
            internals.decoupledStates = methodBuffers.decoupledStates;
            buffers.m_internalBuffers = internals;
        }
        return buffers;
    }
};

template <mean_compatible T> class Mean {
  public:
    using Buffers = MeanBuffers;

  private:
    using Method = std::variant<AtomicMean, DecoupledMean>;

    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const MeanConfig& config) {
        if (std::holds_alternative<AtomicMeanConfig>(config)) {
            const auto methodConfig = std::get<AtomicMeanConfig>(config);
            return AtomicMean(context, shaderCompiler, methodConfig);
        } else if (std::holds_alternative<DecoupledMeanConfig>(config)) {
            const auto methodConfig = std::get<DecoupledMeanConfig>(config);
            return DecoupledMean(context, shaderCompiler, methodConfig);
        }
    }

  public:
    Mean(const merian::ContextHandle& context,
         const merian::ShaderCompilerHandle& shaderCompiler,
         const MeanConfig& config)
        : m_method(createMethod(context, shaderCompiler, config)) {}

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, glsl::uint N) {
        if (std::holds_alternative<AtomicMean>(m_method)) {
            assert(std::holds_alternative<Buffers::AtomicInternals>(buffers.m_internalBuffers));
            AtomicMean::Buffers methodBuffers;
            methodBuffers.elements = buffers.elements;
            methodBuffers.mean = buffers.mean;
            std::get<AtomicMean>(m_method).run(cmd, methodBuffers, N);
        } else if (std::holds_alternative<DecoupledMean>(m_method)) {
            assert(std::holds_alternative<Buffers::DecoupledInternals>(buffers.m_internalBuffers));
            DecoupledMean::Buffers methodBuffers;
            methodBuffers.elements = buffers.elements;
            methodBuffers.mean = buffers.mean;
            methodBuffers.decoupledStates =
                std::get<Buffers::DecoupledInternals>(buffers.m_internalBuffers).decoupledStates;
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};
}; // namespace wrs
