#include "./Pack.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/device/wrs/alias/psa/layout/split.hpp"
#include "src/device/wrs/alias/psa/pack/PackAllocFlags.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <cctype>

device::PackBuffers device::PackBuffers::allocate(merian::ResourceAllocatorHandle alloc,
                                                  merian::MemoryMappingType memoryMapping,
                                                  std::size_t N,
                                                  std::size_t K,
                                                  PackAllocFlags allocFlags) {
    Self buffers;

    if (memoryMapping == merian::MemoryMappingType::NONE) {
        if ((allocFlags & PackAllocFlags::ALLOC_WEIGHTS) != 0) {
            buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                  memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_MEAN) != 0) {
            buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
            buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferDst,
                                                     memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
            buffers.partitionIndices = alloc->createBuffer(
                PartitionIndicesLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_SPLITS) != 0) {
            buffers.splits = device::details::allocateSplitBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, K);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_ALIAS_TABLE) != 0) {
            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, N);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
            buffers.partitionElements = alloc->createBuffer(
                PartitionElementsLayout::size(N),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                memoryMapping);
        }

    } else {

        if ((allocFlags & PackAllocFlags::ALLOC_WEIGHTS) != 0) {
            buffers.weights = alloc->createBuffer(
                WeightsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_MEAN) != 0) {
            buffers.mean = alloc->createBuffer(
                MeanLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
            buffers.heavyCount = alloc->createBuffer(
                HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
            buffers.partitionIndices =
                alloc->createBuffer(PartitionIndicesLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_SPLITS) != 0) {
            buffers.splits = device::details::allocateSplitBuffer(
                alloc, memoryMapping, vk::BufferUsageFlagBits::eTransferSrc, K);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_ALIAS_TABLE) != 0) {
            buffers.aliasTable = device::details::allocateAliasTableBuffer(
                alloc, memoryMapping, vk::BufferUsageFlagBits::eTransferDst, N);
        }

        if ((allocFlags & PackAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
            buffers.partitionElements =
                alloc->createBuffer(PartitionElementsLayout::size(N),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }
    }

    return buffers;
}
