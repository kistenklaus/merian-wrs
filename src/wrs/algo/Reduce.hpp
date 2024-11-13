#pragma once

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/kernels/WorkgroupReduceKernel.hpp"
#include <iostream>
#include <vulkan/vulkan.hpp>

namespace wrs {

class Reduce {
    using weight_t = WorkgroupReduceKernel::weight_t;

  public:
    Reduce(const merian::ContextHandle& context, uint32_t workgroup_size = 512, uint32_t rows = 1)
        : m_workgroupReduceKernel(context, workgroup_size, rows) {}

    merian::BufferHandle run(vk::CommandBuffer cmd,
                             merian::BufferHandle in_weights,
                             merian::BufferHandle out_reduction,
                             std::optional<uint32_t> elementCount = std::nullopt,
                             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        uint32_t currentSize = elementCount.value_or(in_weights->get_size() / sizeof(weight_t));

#ifdef MERIAN_PROFILER_ENABLE
        uint32_t iteration = 0;
        std::optional<merian::ProfileScopeGPU> merian_profile_scope;
        if (profiler) {
            merian_profile_scope = {*profiler, cmd, "record Reduce"};
        }
#endif

        std::array<merian::BufferHandle, 2> pingPong = {in_weights, out_reduction};
        uint32_t pingPongX = 1;
        while (currentSize > 1) {

#ifdef MERIAN_PROFILER_ENABLE
            std::optional<merian::ProfileScopeGPU> merian_profile_scope;
            if (profiler) {
                merian_profile_scope = {*profiler, cmd, fmt ::format("iteration {}", iteration++)};
            }
#endif

            m_workgroupReduceKernel.dispatch(cmd, currentSize, pingPong[pingPongX],
                                             pingPong[pingPongX ^ 1]);

            const auto bar1 = pingPong[pingPongX]->buffer_barrier(vk::AccessFlagBits::eShaderRead,
                                                                  vk::AccessFlagBits::eShaderWrite);
            const auto bar2 = pingPong[pingPongX ^ 1]->buffer_barrier(
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader, {}, {}, {bar1, bar2},
                                {});

            pingPongX ^= 1;
            currentSize /= m_workgroupReduceKernel.getPartitionSize();
        }
        return pingPong[pingPongX];
    }

    vk::DeviceSize requiredResultBufferSize(uint32_t weightCount) {
        return m_workgroupReduceKernel.expectedResultCount(weightCount) * sizeof(weight_t);
    }

  private:
    WorkgroupReduceKernel m_workgroupReduceKernel;
};

}; // namespace wrs
