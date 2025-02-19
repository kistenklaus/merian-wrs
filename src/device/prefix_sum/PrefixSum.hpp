#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/partition/PartitionAllocFlags.hpp"
#include "src/device/prefix_sum/PrefixSumAllocFlags.hpp"
#include "src/device/prefix_sum/block_wise/BlockWiseScan.hpp"
#include "src/device/prefix_sum/decoupled/DecoupledPrefixSum.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include <concepts>
#include <stdexcept>
#include <variant>
namespace device {

using PrefixSumConfig = std::variant<DecoupledPrefixSumConfig, BlockWiseScanConfig>;

[[maybe_unused]]
static std::string prefixSumConfigName(const PrefixSumConfig& config) {
    if (std::holds_alternative<DecoupledPrefixSumConfig>(config)) {
        auto methodConfig = std::get<DecoupledPrefixSumConfig>(config);
        return fmt::format("SingleDispatch-{}-{}-{}",
                           blockScanVariantName(methodConfig.blockScanVariant),
                           methodConfig.workgroupSize, methodConfig.rows);
    } else if (std::holds_alternative<BlockWiseScanConfig>(config)) {
        auto methodConfig = std::get<BlockWiseScanConfig>(config);
        return fmt::format(
            "BlockWise-{}-{}-{}-{}", blockScanVariantName(methodConfig.elementScanConfig.variant),
            methodConfig.elementScanConfig.workgroupSize, methodConfig.elementScanConfig.rows,
            methodConfig.elementScanConfig.sequentialScanLength);
    } else {
        return "UNNAMED";
    }
}

template <typename T>
concept prefix_sum_compatible_type = std::same_as<T, float>;

struct PrefixSumBuffers {
  public:
    using Self = PrefixSumBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    template <prefix_sum_compatible_type T>
    using ElementsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <prefix_sum_compatible_type T>
    using ElementsView = host::layout::BufferView<ElementsLayout<T>>;

    merian::BufferHandle prefixSum;
    template <prefix_sum_compatible_type T>
    using PrefixSumLayout = host::layout::ArrayLayout<T, storageQualifier>;
    template <prefix_sum_compatible_type T>
    using PrefixSumView = host::layout::BufferView<PrefixSumLayout<T>>;

    struct BlockWiseInternals {
        merian::BufferHandle blockScan;
    };

    struct DecoupledInternals {
        merian::BufferHandle decoupledStates;
    };

    std::variant<DecoupledInternals, BlockWiseInternals> m_internalBuffers;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         PrefixSumConfig config,
                         host::glsl::uint N,
                         PrefixSumAllocFlags allocFlags = PrefixSumAllocFlags::ALLOC_ALL) {
        Self buffers;
        if (std::holds_alternative<DecoupledPrefixSumConfig>(config)) {
            const auto instanceConfig = std::get<DecoupledPrefixSumConfig>(config);
            using InstanceBuffers = DecoupledPrefixSum::Buffers;
            auto instanceBuffers = InstanceBuffers::allocate(
                alloc, memoryMapping, N, instanceConfig.partitionSize(), allocFlags);
            buffers.elements = instanceBuffers.elements;
            buffers.prefixSum = instanceBuffers.prefixSum;
            DecoupledInternals internals;
            internals.decoupledStates = instanceBuffers.decoupledStates;
            buffers.m_internalBuffers = internals;
        } else if (std::holds_alternative<BlockWiseScanConfig>(config)) {
            const auto instanceConfig = std::get<BlockWiseScanConfig>(config);
            using InstanceBuffers = BlockWiseScan::Buffers;
            host::glsl::uint blockCount =
                (N + instanceConfig.blockSize() - 1) / instanceConfig.blockSize();
            auto instanceBuffers =
                InstanceBuffers::allocate(alloc, memoryMapping, N, blockCount, allocFlags);
            buffers.elements = instanceBuffers.elements;
            buffers.prefixSum = instanceBuffers.prefixSum;
            BlockWiseInternals internals;
            internals.blockScan = instanceBuffers.reductions;
            buffers.m_internalBuffers = internals;
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
        return buffers;
    }
};

template <prefix_sum_compatible_type T> class PrefixSum {
  private:
    using Method = std::variant<DecoupledPrefixSum, BlockWiseScan>;

    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               PrefixSumConfig config,
                               bool reverseMemoryOrder) {
        if (std::holds_alternative<DecoupledPrefixSumConfig>(config)) {
            const auto& instanceConfig = std::get<DecoupledPrefixSumConfig>(config);
            return DecoupledPrefixSum(context, shaderCompiler, instanceConfig, reverseMemoryOrder);
        } else if (std::holds_alternative<BlockWiseScanConfig>(config)) {
            if (reverseMemoryOrder) {
                throw std::runtime_error("Reversing the memory order of prefix sums is not "
                                         "supported for block-wise prefix scans. Use DecoupledPrefixSum instead");
            }
            const auto& instanceConfig = std::get<BlockWiseScanConfig>(config);
            return BlockWiseScan(context, shaderCompiler, instanceConfig);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    using Buffers = PrefixSumBuffers;
    using Config = PrefixSumConfig;
    using base_type = T;

    PrefixSum(const merian::ContextHandle& context,
              const merian::ShaderCompilerHandle& shaderCompiler,
              PrefixSumConfig config,
              bool reverseMemoryOrder = false)
        : m_method(createMethod(context, shaderCompiler, config, reverseMemoryOrder)) {}

    void run(const merian::CommandBufferHandle& cmd,
             const PrefixSumBuffers& buffers,
             host::glsl::uint N,
             [[maybe_unused]] std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        if (std::holds_alternative<DecoupledPrefixSum>(m_method)) {
            auto method = std::get<DecoupledPrefixSum>(m_method);
            using MethodBuffers = DecoupledPrefixSum::Buffers;
            MethodBuffers methodBuffers;
            methodBuffers.elements = buffers.elements;
            methodBuffers.prefixSum = buffers.prefixSum;
            assert(std::holds_alternative<Buffers::DecoupledInternals>(buffers.m_internalBuffers));
            methodBuffers.decoupledStates =
                std::get<Buffers::DecoupledInternals>(buffers.m_internalBuffers).decoupledStates;
#ifdef MERIAN_PROFILER_ENABLE
            if (profiler.has_value()) {
                std::string label = fmt::format("SingleDispatchPrefixSum [with N={}]", N);
                profiler.value()->start(label);
                profiler.value()->cmd_start(cmd, label);
            }
#endif
            method.run(cmd, methodBuffers, N);
#ifdef MERIAN_PROFILER_ENABLE
            if (profiler.has_value()) {
                profiler.value()->end();
                profiler.value()->cmd_end(cmd);
            }
#endif
        } else if (std::holds_alternative<BlockWiseScan>(m_method)) {
            auto method = std::get<BlockWiseScan>(m_method);
            using MethodBuffers = BlockWiseScan::Buffers;
            MethodBuffers methodBuffers;
            methodBuffers.elements = buffers.elements;
            methodBuffers.prefixSum = buffers.prefixSum;
            assert(std::holds_alternative<Buffers::BlockWiseInternals>(buffers.m_internalBuffers));
            methodBuffers.reductions =
                std::get<Buffers::BlockWiseInternals>(buffers.m_internalBuffers).blockScan;
#ifdef MERIAN_PROFILER_ENABLE
            if (profiler.has_value()) {
                std::string label = fmt::format("BlockWisePrefixSum [with N={}]", N);
                profiler.value()->start(label);
                profiler.value()->cmd_start(cmd, label);
            }
#endif
#ifdef MERIAN_PROFILER_ENABLE
            method.run(cmd, methodBuffers, N, profiler);
#else
            method.run(cmd, methodBuffers, N);
#endif

#ifdef MERIAN_PROFILER_ENABLE
            if (profiler.has_value()) {
                profiler.value()->end();
                profiler.value()->cmd_end(cmd);
            }
#endif
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

    inline host::glsl::uint maxElementCount() const {
        if (std::holds_alternative<DecoupledPrefixSum>(m_method)) {
            return (1 << 28);
        } else {
            return std::get<BlockWiseScan>(m_method).maxElementCount();
        }
    }

  private:
    Method m_method;
};

} // namespace device
