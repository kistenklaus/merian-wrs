#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/wrs/alias/psa/PSA.hpp"
#include "src/device/wrs/alias/sampling/AliasTableSampling.hpp"
#include <fmt/base.h>

namespace device {

struct AliasTableConfig {
    const PSA::Config psaConfig;
    const SampleAliasTable::Config samplingConfig;

    constexpr explicit AliasTableConfig(PSA::Config psaConfig,
                                        SampleAliasTableConfig samplingConfig)
        : psaConfig(psaConfig), samplingConfig(samplingConfig) {}

    inline std::string name() const {
        return fmt::format("AliasTable-{}-Sampling-{}", psaConfig.name(),
                           samplingConfig.workgroupSize);
    }
};

struct AliasTableBuffers {
    using Self = AliasTableBuffers;
    using weight_type = host::glsl::f32;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using SamplesView = host::layout::BufferView<SamplesLayout>;

    merian::BufferHandle m_aliasTable;
    using AliasTableLayout = device::details::AliasTableLayout;
    using AliasTableView = host::layout::BufferView<AliasTableLayout>;

    PSA::Buffers m_psaBuffers;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         const AliasTableConfig config,
                         host::glsl::uint N,
                         host::glsl::uint S) {
        Self buffers;

        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.m_psaBuffers =
                PSA::Buffers::allocate(alloc, memoryMapping, config.psaConfig, N);

            buffers.m_aliasTable = buffers.m_psaBuffers.aliasTable;
            buffers.weights = buffers.m_psaBuffers.weights;

            buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping, "samples");
        } else {
            buffers.weights =
                alloc->createBuffer(WeightsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc,
                                    memoryMapping, "psa-weights");
            buffers.samples =
                alloc->createBuffer(SamplesLayout::size(S), vk::BufferUsageFlagBits::eTransferDst,
                                    memoryMapping, "samples");
            buffers.m_aliasTable = nullptr;
        }

        return buffers;
    }
};

class AliasTable {
  public:
    using Buffers = AliasTableBuffers;
    using Config = AliasTableConfig;

    AliasTable(const merian::ContextHandle& context,
               const merian::ShaderCompilerHandle& shaderCompiler,
               const AliasTableConfig& config)
        : m_psa(context, shaderCompiler, config.psaConfig),
          m_sampling(context, shaderCompiler, config.samplingConfig) {}

    void build(const merian::CommandBufferHandle& cmd,
               const Buffers& buffers,
               host::glsl::uint N,
               std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        m_psa.run(cmd, buffers.m_psaBuffers, N, profiler);

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.m_aliasTable->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                          vk::AccessFlagBits::eShaderRead));
    }

    void sample(const merian::CommandBufferHandle& cmd,
                const Buffers& buffers,
                host::glsl::uint N,
                host::glsl::uint S,
                host::glsl::uint seed) const {
        SampleAliasTable::Buffers samplingBuffers;
        samplingBuffers.aliasTable = buffers.m_aliasTable;
        samplingBuffers.samples = buffers.samples;
        m_sampling.run(cmd, samplingBuffers, N, S, seed);
    }

  private:
    PSA m_psa;
    SampleAliasTable m_sampling;
};

} // namespace device
