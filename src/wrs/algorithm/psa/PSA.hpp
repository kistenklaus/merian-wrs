#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/algorithm/psa/construction/PSAC.hpp"
#include "src/wrs/algorithm/psa/sampling/SampleAliasTable.hpp"
#include "src/wrs/types/glsl.hpp"
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct PSABuffers {
    using Self = PSABuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle weights;
    using WeightsLayout = PSACBuffers::WeightsLayout;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean;
    using MeanLayout = PSACBuffers::MeanLayout;
    using MeanView = layout::BufferView<MeanLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = PSACBuffers::PartitionIndicesLayout;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout = PSACBuffers::PartitionPrefixLayout;
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle partitionDecoupledState;
    using PartitionDecoupledStateLayout = PSACBuffers::PartitionDecoupledStateLayout;
    using PartitionDecoupledStateView = layout::BufferView<PartitionDecoupledStateLayout>;

    merian::BufferHandle splits;
    using SplitsLayout = PSACBuffers::SplitsLayout;
    using SplitsView = layout::BufferView<SplitsLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableLayout = PSACBuffers::AliasTableLayout;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t prefixPartitionSize,
                         std::size_t splitCount,
                         std::size_t S);
};

struct PSAConfig {
    PSACConfig psac;
    glsl::uint samplingWorkgroupSize;

    static constexpr PSAConfig defaultV() {
        return PSAConfig{
            .psac = {},
            .samplingWorkgroupSize = 512,
        };
    }
};

class PSA {
  public:
    using Buffers = PSABuffers;

    explicit PSA(const merian::ContextHandle& context,
                 const merian::ShaderCompilerHandle& shaderCompiler,
                 PSAConfig config = PSAConfig::defaultV())
        : m_psac(context, shaderCompiler, config.psac),
          m_sampleKernel(context, shaderCompiler, config.samplingWorkgroupSize) {}

    void run(const merian::CommandBufferHandle cmd,
             const Buffers& buffers,
             std::size_t N,
             std::size_t S,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {

        PSAC::Buffers psacBuffers;
        psacBuffers.weights = buffers.weights;
        psacBuffers.mean = buffers.mean;
        psacBuffers.partitionPrefix = buffers.partitionPrefix;
        psacBuffers.partitionIndices = buffers.partitionIndices;
        psacBuffers.partitionDecoupledState = buffers.partitionDecoupledState;
        psacBuffers.splits = buffers.splits;
        psacBuffers.aliasTable = buffers.aliasTable;

        if (profiler.has_value()) {
            profiler.value()->start("Construction");
            profiler.value()->cmd_start(cmd, "Construction");
        }
        m_psac.run(cmd, psacBuffers, N, profiler);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     psacBuffers.aliasTable->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eShaderRead));

        SampleAliasTable::Buffers samplingBuffers;
        samplingBuffers.samples = buffers.samples;
        samplingBuffers.aliasTable = buffers.aliasTable;

        if (profiler.has_value()) {
            profiler.value()->start("Sampling");
            profiler.value()->cmd_start(cmd, "Sampling");
        }

        m_sampleKernel.run(cmd, samplingBuffers, N, S);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

    Buffers allocate(const merian::ResourceAllocatorHandle& alloc,
                     const merian::MemoryMappingType& memoryMapping,
                     std::size_t N,
                     std::size_t S) const {
        std::size_t prefixPartitionSize = m_psac.getPrefixPartitionSize();
        std::size_t splitSize = m_psac.getSplitSize();
        std::size_t splitCount = (N + splitSize - 1) / splitSize;
        return Buffers::allocate(alloc, memoryMapping, N, prefixPartitionSize, splitCount, S);
    }

  private:
    PSAC m_psac;
    SampleAliasTable m_sampleKernel;
};

} // namespace wrs
