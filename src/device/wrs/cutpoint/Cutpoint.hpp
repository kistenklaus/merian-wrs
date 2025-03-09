#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 03/03/2025
 * @filename    : Cutpoint.hpp
 *
 * Cutpoint wrs method.
 * Similar to the ITS method, but before sampling constructs a guiding table,
 * which is then used to replace the cooperative narrowing with a simple table lookup.
 */

#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/PrefixSum.hpp"
#include "src/device/wrs/cutpoint/guiding_table/CutpointGuidingTable.hpp"
#include "src/device/wrs/cutpoint/sampling/CutpointSampling.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device {

class CutpointConfig {
  public:
    PrefixSumConfig prefixSumConfig;
    host::glsl::uint guidingEntrySize;

    explicit constexpr CutpointConfig(PrefixSumConfig prefixSumConfig,
                                      host::glsl::uint guidingEntrySize)
        : prefixSumConfig(prefixSumConfig), guidingEntrySize(guidingEntrySize) {}

    std::string name() const {
        return fmt::format("Cutpoint-{}", guidingEntrySize);
    }
};

struct CutpointBuffers {
    using Self = CutpointBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using SamplesView = host::layout::BufferView<SamplesLayout>;

    PrefixSumBuffers m_prefixSumBuffers;

    merian::BufferHandle m_guidingTable;
    using GuidingTableLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;

    merian::BufferHandle m_cmf;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t S,
                         CutpointConfig config) {
        Self buffers;
        buffers.m_prefixSumBuffers =
            PrefixSumBuffers::allocate(alloc, memoryMapping, config.prefixSumConfig, N);
        buffers.weights = buffers.m_prefixSumBuffers.elements;
        buffers.m_cmf = buffers.m_prefixSumBuffers.prefixSum;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);
            std::size_t guidingTableSize =
                (N + config.guidingEntrySize - 1) / config.guidingEntrySize;
            buffers.m_guidingTable = alloc->createBuffer(GuidingTableLayout::size(guidingTableSize),
                                                         vk::BufferUsageFlagBits::eStorageBuffer);
        } else {
            buffers.samples = alloc->createBuffer(
                SamplesLayout::size(S), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

class Cutpoint {
  public:
    using Buffers = CutpointBuffers;
    using Config = CutpointConfig;

    explicit Cutpoint(const merian::ContextHandle& context,
                      const merian::ShaderCompilerHandle& shaderCompiler,
                      Config config)
        : m_scan(context, shaderCompiler, config.prefixSumConfig),
          m_guidingTable(
              context, shaderCompiler, CutpointGuidingTableConfig(512, config.guidingEntrySize)),
          m_sampling(
              context, shaderCompiler, CutpointSamplingConfig(512, config.guidingEntrySize)) {}

    void build(const merian::CommandBufferHandle& cmd,
               const Buffers& buffers,
               host::glsl::uint N,
               std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        if (profiler.has_value()) {
            profiler.value()->start("Prefix-Sum");
            profiler.value()->cmd_start(cmd, "Prefix-Sum");
        }
        m_scan.run(cmd, buffers.m_prefixSumBuffers, N, profiler);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.m_prefixSumBuffers.prefixSum->buffer_barrier(
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead));

        CutpointGuidingTable::Buffers guidingBuffers;
        guidingBuffers.cmf = buffers.m_cmf;
        guidingBuffers.guidingTable = buffers.m_guidingTable;
        if (profiler.has_value()) {
            profiler.value()->start("Guiding-Table");
            profiler.value()->cmd_start(cmd, "Guiding-Table");
        }
        m_guidingTable.run(cmd, guidingBuffers, N);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

    void
    sample(const merian::CommandBufferHandle& cmd,
           const Buffers& buffers,
           host::glsl::uint N,
           host::glsl::uint S,
           host::glsl::uint seed = 12345u,
           [[maybe_unused]] std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        CutpointSampling::Buffers samplingBuffers;
        samplingBuffers.samples = buffers.samples;
        samplingBuffers.cmf = buffers.m_cmf;
        samplingBuffers.guidingTable = buffers.m_guidingTable;
        if (profiler.has_value()) {
            profiler.value()->start("Sampling");
            profiler.value()->cmd_start(cmd, "Sampling");
        }
        m_sampling.run(cmd, samplingBuffers, N, S, seed);
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

  private:
    PrefixSum<host::glsl::f32> m_scan;
    CutpointGuidingTable m_guidingTable;
    CutpointSampling m_sampling;
};

} // namespace device
