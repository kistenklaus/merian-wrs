#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/alias/psa/layout/split.hpp"
#include "src/device/wrs/alias/psa/split/SplitAllocFlags.hpp"
#include "src/device/wrs/alias/psa/split/scalar/ScalarSplit.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <fmt/format.h>
#include <stdexcept>
#include <variant>

namespace device {

using SplitConfig = std::variant<ScalarSplit::Config>;


inline host::glsl::uint splitConfigSplitSize(const SplitConfig& config) {
  if (std::holds_alternative<ScalarSplit::Config>(config)) {
    const auto& methodConfig = std::get<ScalarSplit::Config>(config);
    return methodConfig.splitSize;
  } else {
    throw std::runtime_error("NOT-IMPLEMENTED");
  }
}

inline std::string splitConfigName(const SplitConfig& config) {
  if (std::holds_alternative<ScalarSplit::Config>(config)) {
    const auto& methodConfig = std::get<ScalarSplit::Config>(config);
    return fmt::format("ScalarSplit-{}", methodConfig.splitSize);
  } else {
    throw std::runtime_error("NOT-IMPLEMENTED");
  }
}

struct SplitBuffers {
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;
    using weight_type = host::glsl::f32;
    using Self = SplitBuffers;

    merian::BufferHandle partitionPrefix;
    using PartitionPrefixLayout = host::layout::ArrayLayout<weight_type, storageQualifier>;
    using PartitionPrefixView = host::layout::BufferView<PartitionPrefixLayout>;

    merian::BufferHandle heavyCount;
    using HeavyCountLayout = host::layout::PrimitiveLayout<host::glsl::uint, storageQualifier>;
    using HeavyCountView = host::layout::BufferView<HeavyCountLayout>;

    merian::BufferHandle mean;
    using MeanLayout = host::layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = host::layout::BufferView<MeanLayout>;

    merian::BufferHandle splits;
    using SplitsLayout = device::details::SplitsLayout;
    using SplitsView = host::layout::BufferView<SplitsLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N,
                         host::glsl::uint K,
                         SplitAllocFlags allocFlags = SplitAllocFlags::ALLOC_ALL) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            if ((allocFlags & SplitAllocFlags::ALLOC_PARTITION_PREFIX) != 0) {
                buffers.partitionPrefix = alloc->createBuffer(
                    PartitionPrefixLayout::size(N),
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                    memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
                buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                         vk::BufferUsageFlagBits::eStorageBuffer |
                                                             vk::BufferUsageFlagBits::eTransferDst,
                                                         memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_MEAN) != 0) {
                buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferDst,
                                                   memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_SPLITS) != 0) {
                buffers.splits = device::details::allocateSplitBuffer(
                    alloc, memoryMapping,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                    K);
            }
        } else {
            if ((allocFlags & SplitAllocFlags::ALLOC_PARTITION_PREFIX) != 0) {
                buffers.partitionPrefix =
                    alloc->createBuffer(PartitionPrefixLayout::size(N),
                                        vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
                buffers.heavyCount = alloc->createBuffer(
                    HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_MEAN) != 0) {
                buffers.mean = alloc->createBuffer(
                    MeanLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }
            if ((allocFlags & SplitAllocFlags::ALLOC_SPLITS) != 0) {
                buffers.splits = device::details::allocateSplitBuffer(
                    alloc, memoryMapping, vk::BufferUsageFlagBits::eTransferDst, K);
            }
        }

        return buffers;
    }
};

class Split {
  public:
    using Buffers = SplitBuffers;
    using Config = SplitConfig;

  private:
    using Method = std::variant<ScalarSplit>;

    static Method createMethod(const merian::ContextHandle& context,
                               const merian::ShaderCompilerHandle& shaderCompiler,
                               const Config& config) {
        if (std::holds_alternative<ScalarSplit::Config>(config)) {
            const auto& methodConfig = std::get<ScalarSplit::Config>(config);
            return ScalarSplit(context, shaderCompiler, methodConfig);
        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  public:
    explicit Split(const merian::ContextHandle& context,
                   const merian::ShaderCompilerHandle& shaderCompiler,
                   const Config& config)
        : m_method(createMethod(context, shaderCompiler, config)) {}

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N,
        std::optional<merian::ProfilerHandle> profiler = std::nullopt) const {
        if (std::holds_alternative<ScalarSplit>(m_method)) {
            const auto& method = std::get<ScalarSplit>(m_method);
            ScalarSplit::Buffers methodBuffers;
            methodBuffers.partitionPrefix = buffers.partitionPrefix;
            methodBuffers.heavyCount = buffers.heavyCount;
            methodBuffers.mean = buffers.mean;
            methodBuffers.splits = buffers.splits;
            method.run(cmd, methodBuffers, N, profiler);

        } else {
            throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

    host::glsl::uint splitSize() const {
        if (std::holds_alternative<ScalarSplit>(m_method)) {
            const auto& method = std::get<ScalarSplit>(m_method);
            return method.splitSize();
        } else {
          throw std::runtime_error("NOT-IMPLEMENTED");
        }
    }

  private:
    Method m_method;
};

} // namespace device
