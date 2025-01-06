#include "./DecoupledPrefixSum.hpp"
#include <cstddef>

using Buffers = wrs::DecoupledPrefixSumBuffers;

Buffers Buffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                               merian::MemoryMappingType memoryMapping,
                                               std::size_t N,
                                               std::size_t partitionSize) {

    std::size_t partitionCount = (N + partitionSize - 1) / partitionSize;

    Buffers buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
      buffers.elements = alloc->createBuffer(Buffers::ElementsLayout::size(N),
          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
          merian::MemoryMappingType::NONE);
      buffers.prefixSum = alloc->createBuffer(Buffers::PrefixSumLayout::size(N),
          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
          merian::MemoryMappingType::NONE);
      buffers.decoupledStates = alloc->createBuffer(Buffers::DecoupledStatesLayout::size(partitionCount),
          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
          merian::MemoryMappingType::NONE);
    } else {
      buffers.elements = alloc->createBuffer(Buffers::ElementsLayout::size(N),
          vk::BufferUsageFlagBits::eTransferSrc,
          memoryMapping);
      buffers.prefixSum = alloc->createBuffer(Buffers::PrefixSumLayout::size(N),
          vk::BufferUsageFlagBits::eTransferDst,
          memoryMapping);
      buffers.decoupledStates = alloc->createBuffer(Buffers::DecoupledStatesLayout::size(partitionCount),
          vk::BufferUsageFlagBits::eTransferSrc,
          memoryMapping);
    }
    return buffers;
}
