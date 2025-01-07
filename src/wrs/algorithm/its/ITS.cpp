#include "./ITS.hpp"

wrs::ITSBuffers wrs::ITSBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                          merian::MemoryMappingType memoryMapping,
                                          std::size_t N,
                                          std::size_t S,
                                          std::size_t decoupledPartitionSize) {
    using Buffers = wrs::ITSBuffers;
    Buffers buffers;
    std::size_t partitionCount = (N + decoupledPartitionSize - 1) / decoupledPartitionSize;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.weights = alloc->createBuffer(Buffers::WeightsLayout::size(N),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              merian::MemoryMappingType::NONE);
        buffers.decoupledPrefixState = alloc->createBuffer(
            Buffers::DecoupledPrefixStateLayout::size(partitionCount), 
            vk::BufferUsageFlagBits::eStorageBuffer, merian::MemoryMappingType::NONE);

        buffers.prefixSum = alloc->createBuffer(Buffers::PrefixSumLayout::size(N),
                                                vk::BufferUsageFlagBits::eStorageBuffer,
                                                merian::MemoryMappingType::NONE);
        buffers.samples = alloc->createBuffer(Buffers::SamplesLayout::size(S),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferSrc,
                                              merian::MemoryMappingType::NONE);

    } else {
        buffers.weights = alloc->createBuffer(Buffers::WeightsLayout::size(N),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);

        buffers.decoupledPrefixState = nullptr;
        buffers.prefixSum = nullptr;

        buffers.samples = alloc->createBuffer(Buffers::SamplesLayout::size(S),
                                              vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    }
    return buffers;
}
