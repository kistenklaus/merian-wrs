#pragma once

/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : HS.hpp
 *
 * Hierarchical sampling.
 *
 * Composite of multiple kernels : HSTC, SmallValueOptimization, HSTSampling, Explode
 */

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/hs/explode/Explode.hpp"
#include "src/wrs/algorithm/hs/hstc/HSTC.hpp"
#include "src/wrs/algorithm/hs/sampling/HSTSampling.hpp"
#include "src/wrs/algorithm/hs/svo/SmallValueOptimization.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <fmt/base.h>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct HSBuffers {
    using Self = HSBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle weightTree;
    using WeightTreeLayout = layout::ArrayLayout<float, storageQualifier>;
    using WeightTreeView = layout::BufferView<WeightTreeLayout>;

    merian::BufferHandle outputSensitiveSamples;
    using OutputSensitiveSamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using OutputSensitiveSamplesView = layout::BufferView<OutputSensitiveSamplesLayout>;

    merian::BufferHandle explodeDecoupledStates;
    using _ExplodeDecoupledStateLayout =
        wrs::layout::StructLayout<storageQualifier,
                                  layout::Attribute<glsl::uint, "aggregate">,
                                  layout::Attribute<glsl::uint, "prefix">,
                                  layout::Attribute<glsl::uint, "state">>;
    using _ExplodeDecoupledStateArrayLayout =
        wrs::layout::ArrayLayout<_ExplodeDecoupledStateLayout, storageQualifier>;
    using ExplodeDecoupledStatesLayout = wrs::layout::StructLayout<
        storageQualifier,
        layout::Attribute<glsl::uint, "counter">,
        layout::Attribute<_ExplodeDecoupledStateArrayLayout, "partitions">>;
    using ExplodeDecoupledStatesView = layout::BufferView<ExplodeDecoupledStatesLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t N,
                         std::size_t S,
                         std::size_t explodePartitionSize);
};

class HS {
  public:
    using Buffers = HSBuffers;

    explicit HS(const merian::ContextHandle& context,
                const merian::ShaderCompilerHandle& shaderCompiler,
                glsl::uint hstcWorkgroupSize,
                glsl::uint svoWorkgroupSize,
                glsl::uint samplingWorkgroupSize,
                glsl::uint explodeWorkgroupSize,
                glsl::uint explodeRows,
                glsl::uint explodeLookbackDepth)
        : m_hstc(context, shaderCompiler, hstcWorkgroupSize),
          m_svo(context, shaderCompiler, svoWorkgroupSize),
          m_hstSampling(context, shaderCompiler, samplingWorkgroupSize),
          m_explode(
              context, shaderCompiler, explodeWorkgroupSize, explodeRows, explodeLookbackDepth) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             std::size_t N,
             glsl::uint S,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {

        if (profiler.has_value()) {
            profiler.value()->start("Prepare");
            profiler.value()->cmd_start(cmd, "Prepare");
        }

        const std::size_t explodePartitionSize = m_explode.getPartitionSize();
        const std::size_t explodeWorkgroupCount =
            (N + explodePartitionSize - 1) / explodePartitionSize;
        Buffers::ExplodeDecoupledStatesView localView{buffers.explodeDecoupledStates,
                                                      explodeWorkgroupCount};
        localView.zero(cmd);
        localView.expectComputeRead(cmd);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        if (profiler.has_value()) {
            profiler.value()->start("Construction");
            profiler.value()->cmd_start(cmd, "Construction");
        }

        wrs::HSTC::Buffers hstcBuffers;
        hstcBuffers.tree = buffers.weightTree;
        m_hstc.run(cmd, hstcBuffers, N, m_svo.getWorkgroupSize());

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.weightTree->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                        vk::AccessFlagBits::eShaderRead));

        HSSVOBuffers svoBuffers;
        svoBuffers.hst = buffers.weightTree;
        svoBuffers.histogram = buffers.outputSensitiveSamples;
        wrs::hst::HSTRepr repr{N};
        glsl::uint offset = 0;
        glsl::uint size = N;
        for (const auto& level : repr.get()) {
            if (level.numChildren <= m_svo.getWorkgroupSize()) {
                offset = level.childOffset;
                size = level.numChildren;
                break;
            }
        }

        if (profiler.has_value()) {
            profiler.value()->start("SVO");
            profiler.value()->cmd_start(cmd, "SVO");
        }

        m_svo.run(cmd, svoBuffers, S, offset, size);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.outputSensitiveSamples->buffer_barrier(
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead));

        if (profiler.has_value()) {
            profiler.value()->start("Sampling");
            profiler.value()->cmd_start(cmd, "Sampling");
        }

        wrs::HSTSampling::Buffers samplingBuffers;
        samplingBuffers.hst = buffers.weightTree;
        samplingBuffers.samples = buffers.outputSensitiveSamples;
        m_hstSampling.run(cmd, samplingBuffers, N, m_svo.getWorkgroupSize());

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.outputSensitiveSamples->buffer_barrier(
                         vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead));

        if (profiler.has_value()) {
            profiler.value()->start("Explode");
            profiler.value()->cmd_start(cmd, "Explode");
        }

        wrs::Explode::Buffers explodeBuffers;
        explodeBuffers.outputSensitive = buffers.outputSensitiveSamples;
        explodeBuffers.samples = buffers.samples;
        explodeBuffers.decoupledState = buffers.explodeDecoupledStates;
        m_explode.run(cmd, explodeBuffers, N);

        if (profiler.has_value()) {
            profiler.value()->end();
            profiler.value()->cmd_end(cmd);
        }
    }

  private:
    wrs::HSTC m_hstc;
    wrs::HSSVO m_svo;
    wrs::HSTSampling m_hstSampling;
    wrs::Explode m_explode;
};

} // namespace wrs
