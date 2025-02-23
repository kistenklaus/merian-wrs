#include "./SplitPack.hpp"

device::SplitPackBuffers::Self
device::SplitPackBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                   merian::MemoryMappingType memoryMapping,
                                   const SplitPackConfig config,
                                   host::glsl::uint N,
                                   SplitPackAllocFlags allocFlags) {
    Self buffers;

    if (memoryMapping == merian::MemoryMappingType::NONE) {
        if ((allocFlags & SplitPackAllocFlags::ALLOC_WEIGHTS) != 0) {
            buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                  memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_MEAN) != 0) {
            buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
            buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferDst,
                                                     memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
            buffers.partitionIndices = alloc->createBuffer(
                PartitionIndicesLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_ALIAS_TABLE) != 0) {
            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, N);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
            buffers.partitionElements = alloc->createBuffer(
                PartitionElementsLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                memoryMapping);
        }

    } else {

        if ((allocFlags & SplitPackAllocFlags::ALLOC_WEIGHTS) != 0) {
            buffers.weights = alloc->createBuffer(
                WeightsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_MEAN) != 0) {
            buffers.mean = alloc->createBuffer(
                MeanLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
            buffers.heavyCount = alloc->createBuffer(
                HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
            buffers.partitionIndices =
                alloc->createBuffer(PartitionIndicesLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_ALIAS_TABLE) != 0) {
            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping, vk::BufferUsageFlagBits::eTransferDst, N);
        }

        if ((allocFlags & SplitPackAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
            buffers.partitionElements =
                alloc->createBuffer(PartitionElementsLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }
    }

    if (std::holds_alternative<SerialSplitPack::Config>(config)) {
        const auto& methodConfig = std::get<SerialSplitPack::Config>(config);
        const auto splitSize = splitConfigSplitSize(methodConfig.splitConfig);
        const auto K = (N + splitSize - 1) / splitSize;
        SerialInternals internals;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            internals.splits = device::details::allocateSplitBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, K);
        } else {
            internals.splits = device::details::allocateSplitBuffer(
                alloc, memoryMapping, vk::BufferUsageFlagBits::eTransferDst, K);
        }

        buffers.m_internals = internals;
    } else if (std::holds_alternative<InlineSplitPack::Config>(config)) {
        InlineInternals internals;
        buffers.m_internals = internals;
    } else {
        throw std::runtime_error("NOT-IMPLEMNETED");
    }
    return buffers;
}

