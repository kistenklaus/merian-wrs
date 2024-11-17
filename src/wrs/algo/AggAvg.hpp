#pragma once

#include "src/wrs/algo/Reduce.hpp"
#include "src/wrs/kernels/FinalAggAvgKernel.hpp"
#include <vulkan/vulkan_enums.hpp>
namespace wrs {

class AggAvg {
    using value_t = float;

  public:
    AggAvg(const merian::ContextHandle& context) : m_reduceAlgo(context), m_aggAvgKernel(context) {}

    merian::BufferHandle run(vk::CommandBuffer cmd,
                             merian::BufferHandle elements,
                             merian::BufferHandle meta,
                             std::optional<uint32_t> elementCount = std::nullopt,
                             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {

#ifdef MERIAN_PROFILER_ENABLE
        std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
        if (profiler.has_value()) {
            merian_profile_scope.emplace(profiler.value(), cmd,
                                         fmt::format("reduce and average (AggAvg)"));
        }
#endif

        uint32_t count = elementCount.value_or(elements->get_size() / sizeof(value_t));
        merian::BufferHandle agg = m_reduceAlgo.run(cmd, elements, meta, count, profiler);

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
            {}, {},
            agg->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead),
            {});

        {
#ifdef MERIAN_PROFILER_ENABLE
            std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
            if (profiler.has_value()) {
                merian_profile_scope.emplace(profiler.value(), cmd,
                                             fmt::format("compute average"));
            }
#endif
            m_aggAvgKernel.dispatch(cmd, agg, count);
        }

        return agg;
    }

    vk::DeviceSize minMetaBufferSize(uint32_t maxElements) {
        return std::max(m_reduceAlgo.requiredResultBufferSize(maxElements), sizeof(float) * 2);
    }

  private:
    Reduce m_reduceAlgo;
    FinalAggAvgKernel m_aggAvgKernel;
};

} // namespace wrs
