#include "./SubgroupPack.hpp"

wrs::SubgroupPackBuffers wrs::SubgroupPackBuffers::allocate(merian::ResourceAllocatorHandle alloc,
                                  std::size_t weightCount,
                                  std::size_t splitCount,
                                  merian::MemoryMappingType memoryMapping) {
    SubgroupPackBuffers buffers;
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
        buffers.partition = alloc->createBuffer(PartitionLayout::size(weightCount),
                vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst,
                memoryMapping);
        buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
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

        buffers.partition = alloc->createBuffer(PartitionLayout::size(weightCount),
                vk::BufferUsageFlagBits::eTransferSrc,
                memoryMapping);

        buffers.partitionPrefix = alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
            vk::BufferUsageFlagBits::eTransferSrc,
            memoryMapping);
    }
    return buffers;
}
