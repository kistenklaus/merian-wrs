#include "./PSA.hpp"

wrs::PSABuffers wrs::PSABuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                          merian::MemoryMappingType memoryMapping,
                                          std::size_t N,
                                          std::size_t meanPartitionSize,
                                          std::size_t prefixPartitionSize,
                                          std::size_t splitCount,
                                          std::size_t S) {
    wrs::PSABuffers buffers;
    std::size_t meanWorkgroupCount = (N + meanPartitionSize - 1) / meanPartitionSize;
    std::size_t prefixWorkgroupCount = (N + prefixPartitionSize - 1) / prefixPartitionSize;

    if (memoryMapping == merian::MemoryMappingType::NONE) {

        buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
        buffers.meanDecoupledStates =
            alloc->createBuffer(MeanDecoupledStatesLayout::size(meanWorkgroupCount),
                                vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                           vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
        buffers.partitionIndices =
            alloc->createBuffer(PartitionIndicesLayout::size(N),
                                vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
        buffers.partitionPrefix =
            alloc->createBuffer(PartitionPrefixLayout::size(N),
                                vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
         buffers.partitionDecoupledState = 
             alloc->createBuffer(PartitionDecoupledStateLayout::size(prefixWorkgroupCount), 
                                 vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping); 
         buffers.splits = alloc->createBuffer( 
             SplitLayout::size(splitCount), vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping); 
         buffers.aliasTable = alloc->createBuffer(AliasTableLayout::size(N), 
                                                  vk::BufferUsageFlagBits::eStorageBuffer,
                                                  memoryMapping); 
        buffers.samples = alloc->createBuffer(
            SamplesLayout::size(S), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            memoryMapping);
    } else {
        buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.meanDecoupledStates = nullptr;
        buffers.mean = nullptr;
        buffers.partitionIndices = nullptr;
        buffers.partitionPrefix = nullptr;
        buffers.partitionDecoupledState = nullptr;
        buffers.splits = nullptr;
        buffers.aliasTable = nullptr;
        buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                              vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    }
    return buffers;
}
