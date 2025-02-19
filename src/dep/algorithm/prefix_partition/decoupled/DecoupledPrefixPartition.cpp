
#include "./DecoupledPrefixPartition.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "vulkan/vulkan_enums.hpp"

template <>
wrs::DecoupledPrefixPartitionBuffers
wrs::DecoupledPrefixPartitionBuffers::allocate<float>(const merian::ResourceAllocatorHandle& alloc,
                                                      merian::MemoryMappingType memoryMapping,
                                                      std::size_t N,
                                                      std::size_t blockCount) {
    using T = float;
    Self buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.elements = alloc->createBuffer(ElementsLayout<T>::size(N),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               memoryMapping);
        buffers.pivot = alloc->createBuffer(PivotLayout<T>::size(),
                                            vk::BufferUsageFlagBits::eStorageBuffer |
                                                vk::BufferUsageFlagBits::eTransferDst,
                                            memoryMapping);
        buffers.decoupledStates = alloc->createBuffer(DecoupledStatesLayout<T>::size(blockCount),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        buffers.partitionIndices = alloc->createBuffer(PartitionIndicesLayout::size(N),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferSrc,
                                                       memoryMapping);
        buffers.partitionElements = alloc->createBuffer(PartitionElementsLayout<T>::size(N),
                                                        vk::BufferUsageFlagBits::eStorageBuffer |
                                                            vk::BufferUsageFlagBits::eTransferSrc,
                                                        memoryMapping);
        buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                                          vk::BufferUsageFlagBits::eTransferSrc,
                                                      memoryMapping);
        buffers.heavyCount = alloc->createBuffer(HeavyCountLayout<T>::size(),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferSrc,
                                                 memoryMapping);
    } else {
        buffers.elements = alloc->createBuffer(
            ElementsLayout<T>::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.pivot = alloc->createBuffer(PivotLayout<T>::size(),
                                            vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.decoupledStates = nullptr;
        buffers.partitionIndices = alloc->createBuffer(
            PartitionIndicesLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.partitionElements =
            alloc->createBuffer(PartitionElementsLayout<T>::size(N),
                                vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.partitionPrefix =
            alloc->createBuffer(PartitionPrefixLayout<T>::size(N),
                                vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.heavyCount = alloc->createBuffer(
            HeavyCountLayout<T>::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    }

    return buffers;
}
