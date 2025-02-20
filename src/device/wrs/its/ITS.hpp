#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : ITS.hpp
 *
 * Inverse Transform Sampling method.
 * First computes the cummulative distribution (weight) function (CDF/CMF)
 * and then performs a the inverse CDF method, which boils down to a
 * uniform sample and a binary search.
 *
 * This method performs suprisingly good for small sample sizes.
 */

#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/PrefixSum.hpp"
#include "src/device/wrs/its/sampling/InverseTransformSampling.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct ITSBuffers {
    using Self = ITSBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using SamplesView = host::layout::BufferView<SamplesLayout>;

    PrefixSumBuffers m_prefixSumBuffers;
    InverseTransformSamplingBuffers m_samplingBuffers;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t S,
                         PrefixSumConfig prefixSumConfig) {
        Self buffers;
        buffers.m_prefixSumBuffers =
            PrefixSumBuffers::allocate(alloc, memoryMapping, prefixSumConfig, N);
        buffers.weights = buffers.m_prefixSumBuffers.elements;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);
        } else {
            buffers.samples = alloc->createBuffer(
                SamplesLayout::size(S), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        buffers.m_samplingBuffers.samples = buffers.samples;
        buffers.m_samplingBuffers.cmf = buffers.m_prefixSumBuffers.prefixSum;

        return buffers;
    }
};

class ITSConfig {
  public:
    PrefixSumConfig prefixSumConfig;
    InverseTransformSamplingConfig samplingConfig;

    constexpr ITSConfig() : prefixSumConfig{}, samplingConfig{} {}
    explicit constexpr ITSConfig(PrefixSumConfig prefixSumConfig,
                                 InverseTransformSamplingConfig samplingConfig)
        : prefixSumConfig(prefixSumConfig), samplingConfig(samplingConfig) {}
};

class ITS {
  public:
    using Buffers = ITSBuffers;
    using Config = ITSConfig;

    explicit ITS(const merian::ContextHandle& context,
                 const merian::ShaderCompilerHandle& shaderCompiler,
                 ITSConfig config = {})
        : m_prefixSumKernel(context, shaderCompiler, config.prefixSumConfig),
          m_samplingKernel(context, shaderCompiler, config.samplingConfig) {}

    void build(const merian::CommandBufferHandle& cmd,
               const Buffers& buffers,
               host::glsl::uint N,
               std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        if (profiler.has_value()) {
            profiler.value()->start("Prefix Sum");
            profiler.value()->cmd_start(cmd, "Prefix Sum");
        }
        m_prefixSumKernel.run(cmd, buffers.m_prefixSumBuffers, N, profiler);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.m_prefixSumBuffers.prefixSum->buffer_barrier(
                         vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite));
    }

    void
    sample(const merian::CommandBufferHandle& cmd,
           const Buffers& buffers,
           host::glsl::uint N,
           host::glsl::uint S,
           host::glsl::uint seed = 12345u,
           [[maybe_unused]] std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        using SamplingBuffers = InverseTransformSampling::Buffers;
        SamplingBuffers samplingBuffers;
        samplingBuffers.cmf = buffers.m_prefixSumBuffers.prefixSum;
        samplingBuffers.samples = buffers.samples;
        if (profiler.has_value()) {
            profiler.value()->start("Sampling");
            profiler.value()->cmd_start(cmd, "Sampling");
        }
        m_samplingKernel.run(cmd, samplingBuffers, N, S, seed);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

  private:
    PrefixSum<host::glsl::f32> m_prefixSumKernel;
    InverseTransformSampling m_samplingKernel;
};

} // namespace device
