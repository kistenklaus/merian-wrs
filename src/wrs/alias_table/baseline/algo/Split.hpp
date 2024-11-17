#pragma once

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/alias_table/baseline/kernels/SplitKernel.hpp"
namespace wrs::baseline {

class Split {
  public:
    Split(const merian::ContextHandle& context) : m_splitKernel(context) {}

    void run(vk::CommandBuffer cmd,
             merian::BufferHandle in_weights,
             merian::BufferHandle in_avgPrefixSum,
             merian::BufferHandle in_lightHeavy,
             merian::BufferHandle in_lightHeavyPrefix,
             merian::BufferHandle out_splitInfo,
             uint32_t splitCount,
             std::optional<uint32_t> weightCount = std::nullopt,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        uint32_t count = weightCount.value_or(in_weights->get_size() / sizeof(float));

#ifdef MERIAN_PROFILER_ENABLE
        std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
        if (profiler.has_value()) {
            merian_profile_scope.emplace(profiler.value(), cmd,
                                         fmt::format("split [count = {}, splitCount = {}]", count, splitCount));
        }
#endif

        m_splitKernel.dispatch(cmd, in_weights, count, in_avgPrefixSum, in_lightHeavy,
                               in_lightHeavyPrefix, out_splitInfo, splitCount);
    }

    vk::DeviceSize splitDescriptorSize() {
      return m_splitKernel.splitDescriptorSize();
    }

  private:
    baseline::SplitKernel m_splitKernel;
};

} // namespace wrs::baseline
