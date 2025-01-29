#include "./PSAC.hpp"

wrs::PSACBuffers wrs::PSACBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                 const std::size_t weightCount,
                                 const std::size_t meanPartitionSize,
                                 const std::size_t prefixPartitionSize,
                                 const std::size_t splitCount,
                                 const merian::MemoryMappingType memoryMapping) {
    PSACBuffers buffers;
    std::size_t prefixWorkgroupCount = (weightCount + prefixPartitionSize - 1) / prefixPartitionSize;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.weights = alloc->createBuffer(WeightsLayout::size(weightCount),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                           vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
        buffers.partitionIndices =
            alloc->createBuffer(PartitionIndicesLayout::size(weightCount),
                                vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
        buffers.partitionPrefix =
            alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
                                vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
         buffers.partitionDecoupledState = 
             alloc->createBuffer(PartitionDecoupledStateLayout::size(prefixWorkgroupCount), 
                                 vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping); 
         buffers.splits = alloc->createBuffer( 
             SplitsLayout::size(splitCount), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, memoryMapping); 
         buffers.aliasTable = alloc->createBuffer(AliasTableLayout::size(weightCount), 
                                                  vk::BufferUsageFlagBits::eStorageBuffer | 
                                                      vk::BufferUsageFlagBits::eTransferSrc, 
                                                  memoryMapping); 
    } else {
        buffers.weights = alloc->createBuffer(WeightsLayout::size(weightCount),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(), {}, memoryMapping);
        buffers.partitionIndices =
            alloc->createBuffer(PartitionIndicesLayout::size(weightCount), {}, memoryMapping);
        buffers.partitionPrefix =
            alloc->createBuffer(PartitionPrefixLayout::size(weightCount), {}, memoryMapping);
         buffers.partitionDecoupledState = 
             alloc->createBuffer(PartitionDecoupledStateLayout::size(prefixWorkgroupCount), {}, memoryMapping); 
         buffers.splits = alloc->createBuffer(SplitsLayout::size(splitCount), vk::BufferUsageFlagBits::eTransferDst, memoryMapping); 
         buffers.aliasTable = 
             alloc->createBuffer(AliasTableLayout::size(weightCount), 
                                 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping); 
    }
    return buffers;
}
