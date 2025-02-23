#include "./ScalarPack.hpp"

device::ScalarPackBuffers device::ScalarPackBuffers::allocate(merian::ResourceAllocatorHandle alloc,
                                  std::size_t weightCount,
                                  std::size_t splitCount,
                                  merian::MemoryMappingType memoryMapping) {
    ScalarPackBuffers buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.partitionIndices = alloc->createBuffer(PartitionIndicesLayout::size(weightCount),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       memoryMapping);
        buffers.weights = alloc->createBuffer(WeightsLayout::size(weightCount),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                           vk::BufferUsageFlagBits::eStorageBuffer |
                                               vk::BufferUsageFlagBits::eTransferDst,
                                           memoryMapping);
        buffers.splits = alloc->createBuffer(SplitsLayout::size(splitCount),
                                             vk::BufferUsageFlagBits::eStorageBuffer |
                                                 vk::BufferUsageFlagBits::eTransferDst,
                                             memoryMapping);
        buffers.aliasTable = alloc->createBuffer(AliasTableLayout::size(weightCount),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferSrc,
                                                 memoryMapping);
    } else {
        buffers.partitionIndices =
            alloc->createBuffer(PartitionIndicesLayout::size(weightCount),
                                vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.weights = alloc->createBuffer(WeightsLayout::size(weightCount),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                           vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.splits = alloc->createBuffer(SplitsLayout::size(splitCount),
                                             vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.aliasTable =
            alloc->createBuffer(AliasTableLayout::size(weightCount),
                                vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    }
    return buffers;
}
