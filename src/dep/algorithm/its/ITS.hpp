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
 *
 */

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/its/sampling/InverseTransformSampling.hpp"
#include "src/wrs/algorithm/prefix_sum/decoupled/DecoupledPrefixSum.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/glsl.hpp"
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct ITSBuffers {
    using Self = ITSBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle decoupledPrefixState;
    using DecoupledPrefixStateLayout = wrs::DecoupledPrefixSum::Buffers::DecoupledStatesLayout;
    using DecoupledPrefixStateView = layout::BufferView<DecoupledPrefixStateLayout>;

    merian::BufferHandle prefixSum;
    using PrefixSumLayout = layout::ArrayLayout<float, storageQualifier>;
    using PrefixSumView = layout::BufferView<PrefixSumLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t S,
                         std::size_t decoupledPartitionSize);
};

class ITSConfig {
  public:
    DecoupledPrefixSumConfig prefixSumConfig;
    InverseTransformSamplingConfig samplingConfig;

    constexpr ITSConfig() : prefixSumConfig{}, samplingConfig{} {}
    explicit constexpr ITSConfig(DecoupledPrefixSumConfig prefixSumConfig,
                                 InverseTransformSamplingConfig samplingConfig)
        : prefixSumConfig(prefixSumConfig), samplingConfig(samplingConfig) {}
};

class ITS {
  public:
    using Buffers = ITSBuffers;

    explicit ITS(const merian::ContextHandle& context,
                 const merian::ShaderCompilerHandle& shaderCompiler,
                 ITSConfig config = {})
        : m_prefixSumKernel(context, shaderCompiler, config.prefixSumConfig),
          m_samplingKernel(context, shaderCompiler, config.samplingConfig) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             glsl::uint N,
             glsl::uint S,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        using PrefixBuffers = DecoupledPrefixSum::Buffers;
        PrefixBuffers prefixBuffers;
        prefixBuffers.elements = buffers.weights;
        prefixBuffers.prefixSum = buffers.prefixSum;
        prefixBuffers.decoupledStates = buffers.decoupledPrefixState;

        std::size_t prefixPartitionSize = m_prefixSumKernel.getPartitionSize();
        std::size_t prefixPartitionCount = (N + prefixPartitionSize - 1) / prefixPartitionSize;

        PrefixBuffers::DecoupledStatesView decoupledStateView{buffers.decoupledPrefixState,
                                                              prefixPartitionCount};

        if (profiler.has_value()) {
            profiler.value()->start("Prepare");
            profiler.value()->cmd_start(cmd, "Prepare");
        }
        decoupledStateView.zero(cmd);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
        decoupledStateView.expectComputeRead(cmd);

        if (profiler.has_value()) {
            profiler.value()->start("Prefix Sum");
            profiler.value()->cmd_start(cmd, "Prefix Sum");
        }
        m_prefixSumKernel.run(cmd, prefixBuffers, N);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.prefixSum->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                       vk::AccessFlagBits::eShaderRead));
        using SamplingBuffers = InverseTransformSampling::Buffers;
        SamplingBuffers samplingBuffers;
        samplingBuffers.cmf = buffers.prefixSum;
        samplingBuffers.samples = buffers.samples;

        if (profiler.has_value()) {
            profiler.value()->start("Sampling");
            profiler.value()->cmd_start(cmd, "Sampling");
        }
        m_samplingKernel.run(cmd, samplingBuffers, N, S);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

  private:
    DecoupledPrefixSum m_prefixSumKernel;
    InverseTransformSampling m_samplingKernel;
};

} // namespace wrs
