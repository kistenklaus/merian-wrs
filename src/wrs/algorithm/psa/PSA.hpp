#pragma once

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
    using WeightsLayout = layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle meanDecoupledStates;
    using MeanDecoupledStatesLayout = DecoupledMeanBuffers::DecoupledStatesLayout;
    using MeanDecoupledStateView = layout::BufferView<MeanDecoupledStatesLayout>;

    merian::BufferHandle mean;
    using MeanLayout = layout::PrimitiveLayout<float, storageQualifier>;
    using MeanView = layout::BufferView<MeanLayout>;

    merian::BufferHandle partitionIndices;
    using PartitionIndicesLayout = DecoupledPrefixPartitionBuffers::PartitionLayout;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout = DecoupledPrefixPartitionBuffers::PartitionPrefixLayout;
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle partitionDecoupledState;
    using PartitionDecoupledStateLayout = DecoupledPrefixPartitionBuffers::BatchDescriptorsLayout;
    using PartitionDecoupledStateView = layout::BufferView<PartitionDecoupledStateLayout>;

    merian::BufferHandle splits;
    using SplitLayout = ScalarSplitBuffers::SplitsLayout;
    using SplitView = layout::BufferView<SplitLayout>;

    merian::BufferHandle aliasTable;
    using AliasTableLayout = ScalarPackBuffers::AliasTableLayout;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

   static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                                          merian::MemoryMappingType memoryMapping,
                                          std::size_t N,
                                          std::size_t meanPartitionSize,
                                          std::size_t prefixPartitionSize,
                                          std::size_t splitCount,
                                          std::size_t S);
};

struct PSAConfig {
    PSACConfig psac;
    glsl::uint samplingWorkgroupSize;
    glsl::uint splitSize;

    static constexpr PSAConfig defaultV() {
      return PSAConfig {
        .psac = {},
        .samplingWorkgroupSize = 512,
        .splitSize = 32,
      };
    }
};

class PSA {
  public:
    using Buffers = PSABuffers;

    explicit PSA(const merian::ContextHandle& context, PSAConfig config = PSAConfig::defaultV())
        : m_psac(context, config.psac), m_sampleKernel(context, config.samplingWorkgroupSize),
          m_splitSize(config.splitSize) {}

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, std::size_t N, std::size_t S,
        std::optional<merian::ProfilerHandle> profiler = std::nullopt) {

        std::size_t splitCount = N / m_splitSize;

        PSAC::Buffers psacBuffers;

        if (profiler.has_value()) {
          profiler.value()->start("Construction");
          profiler.value()->cmd_start(cmd, "Construction");
        }
        m_psac.run(cmd, psacBuffers, N, splitCount);

        if (profiler.has_value()) {
          profiler.value()->end();
          profiler.value()->cmd_end(cmd);
        }

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                            psacBuffers.aliasTable->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                   vk::AccessFlagBits::eShaderRead),
                            {});

        SampleAliasTable::Buffers samplingBuffers;

        if (profiler.has_value()) {
          profiler.value()->start("Sampling");
          profiler.value()->cmd_start(cmd,"Sampling");
        }
        m_sampleKernel.run(cmd, samplingBuffers, N, S);

        if (profiler.has_value()) {
          profiler.value()->end();
          profiler.value()->cmd_end(cmd);
        }
    }

  private:
    PSAC m_psac;
    SampleAliasTable m_sampleKernel;
    glsl::uint m_splitSize;
};

} // namespace wrs
