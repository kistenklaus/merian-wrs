
#include "./DecoupledPrefixPartition.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "vulkan/vulkan_enums.hpp"

template <>
device::DecoupledPrefixPartitionBuffers
device::DecoupledPrefixPartitionBuffers::allocate<float>(const merian::ResourceAllocatorHandle& alloc,
                                                      merian::MemoryMappingType memoryMapping,
                                                      std::size_t N,
                                                      std::size_t blockCount,
                                                      PrefixPartitionAllocFlags allocFlags) {
    using T = float;
    Self buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
          buffers.elements = alloc->createBuffer(ElementsLayout<T>::size(N),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferDst,
                                                 memoryMapping);
        }
        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PIVOT) != 0) {
          buffers.pivot = alloc->createBuffer(PivotLayout<T>::size(),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
        }
        buffers.decoupledStates = alloc->createBuffer(DecoupledStatesLayout<T>::size(blockCount),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
          buffers.partitionIndices = alloc->createBuffer(PartitionIndicesLayout::size(N),
                                                         vk::BufferUsageFlagBits::eStorageBuffer |
                                                             vk::BufferUsageFlagBits::eTransferSrc,
                                                         memoryMapping);
        }
        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
          buffers.partitionElements = alloc->createBuffer(PartitionElementsLayout<T>::size(N),
                                                          vk::BufferUsageFlagBits::eStorageBuffer |
                                                              vk::BufferUsageFlagBits::eTransferSrc,
                                                          memoryMapping);
        }

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_PREFIX) != 0) {
          buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                                        vk::BufferUsageFlagBits::eStorageBuffer |
                                                            vk::BufferUsageFlagBits::eTransferSrc,
                                                        memoryMapping);
        }
        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
          buffers.heavyCount = alloc->createBuffer(HeavyCountLayout<T>::size(),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferSrc,
                                                   memoryMapping);
        }
    } else {

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
          buffers.elements = alloc->createBuffer(
              ElementsLayout<T>::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PIVOT) != 0) {
          buffers.pivot = alloc->createBuffer(PivotLayout<T>::size(),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }
        buffers.decoupledStates = nullptr;

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
          buffers.partitionIndices = alloc->createBuffer(
              PartitionIndicesLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
          buffers.partitionElements =
              alloc->createBuffer(PartitionElementsLayout<T>::size(N),
                                  vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_PARTITION_PREFIX) != 0) {
          buffers.partitionPrefix =
              alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                  vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }

        if ((allocFlags & PrefixPartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
          buffers.heavyCount = alloc->createBuffer(
              HeavyCountLayout<T>::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
    }

    return buffers;
}
