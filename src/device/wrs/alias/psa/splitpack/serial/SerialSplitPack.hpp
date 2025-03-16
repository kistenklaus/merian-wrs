#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/alias/psa/pack/Pack.hpp"
#include "src/device/wrs/alias/psa/split/Split.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device {

struct SerialSplitPackBuffers {
    using Self = SerialSplitPackBuffers;

    merian::BufferHandle weights;

    merian::BufferHandle partitionIndices;

    merian::BufferHandle partitionPrefix;

    merian::BufferHandle heavyCount;

    merian::BufferHandle mean;

    merian::BufferHandle aliasTable;

    merian::BufferHandle splits;

    merian::BufferHandle partitionElements; // optional
};

struct SerialSplitPackConfig {
    SplitConfig splitConfig;
    PackConfig packConfig;

    explicit SerialSplitPackConfig(SplitConfig splitConfig, PackConfig packConfig)
        : splitConfig(splitConfig), packConfig(packConfig) {
        assert(splitConfigSplitSize(splitConfig) == packConfigSplitSize(packConfig));
    }
};

class SerialSplitPack {
  public:
    using Buffers = SerialSplitPackBuffers;
    using Config = SerialSplitPackConfig;

    SerialSplitPack(const merian::ContextHandle& context,
                    const merian::ShaderCompilerHandle& shaderCompiler,
                    const Config& config,
                    bool usePartitionElements)
        : m_split(context, shaderCompiler, config.splitConfig),
          m_pack(context, shaderCompiler, config.packConfig, usePartitionElements) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {


        Split::Buffers splitBuffers;
        splitBuffers.partitionPrefix = buffers.partitionPrefix;
        splitBuffers.heavyCount = buffers.heavyCount;
        splitBuffers.mean = buffers.mean;
        splitBuffers.splits = buffers.splits;

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("Split");
            profiler.value()->cmd_start(cmd, "Split");
        }
#endif
        m_split.run(cmd, splitBuffers, N, profiler);

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.splits->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                    vk::AccessFlagBits::eShaderRead));

        Pack::Buffers packBuffers;
        packBuffers.mean = buffers.mean;
        packBuffers.splits = buffers.splits;
        packBuffers.heavyCount = buffers.heavyCount;
        packBuffers.partitionElements = buffers.partitionElements;
        packBuffers.partitionIndices = buffers.partitionIndices;
        packBuffers.aliasTable = buffers.aliasTable;
        packBuffers.weights = buffers.weights;

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->start("Pack");
            profiler.value()->cmd_start(cmd, "Pack");
        }
#endif

        m_pack.run(cmd, packBuffers, N, profiler);

#ifdef MERIAN_PROFILER_ENABLE
        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
#endif
    }

  private:
    Split m_split;
    Pack m_pack;
};

} // namespace device
