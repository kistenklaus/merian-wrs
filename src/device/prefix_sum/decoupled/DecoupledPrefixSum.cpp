#include "./DecoupledPrefixSum.hpp"
#include <cstddef>

using Buffers = device::DecoupledPrefixSumBuffers;

Buffers Buffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                               merian::MemoryMappingType memoryMapping,
                                               std::size_t N,
                                               std::size_t partitionSize,
                                               PrefixSumAllocFlags allocFlags) {

    std::size_t partitionCount = (N + partitionSize - 1) / partitionSize;

    Buffers buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
      if ((allocFlags & PrefixSumAllocFlags::ALLOC_ELEMENTS) != 0) {
        buffers.elements = alloc->createBuffer(Buffers::ElementsLayout::size(N),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            merian::MemoryMappingType::NONE);
      }
      if ((allocFlags & PrefixSumAllocFlags::ALLOC_PREFIX_SUM) != 0) {
        buffers.prefixSum = alloc->createBuffer(Buffers::PrefixSumLayout::size(N),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            merian::MemoryMappingType::NONE);
      }
      buffers.decoupledStates = alloc->createBuffer(Buffers::DecoupledStatesLayout::size(partitionCount),
          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
          merian::MemoryMappingType::NONE);
    } else {
      if ((allocFlags & PrefixSumAllocFlags::ALLOC_ELEMENTS) != 0) {
        buffers.elements = alloc->createBuffer(Buffers::ElementsLayout::size(N),
            vk::BufferUsageFlagBits::eTransferSrc,
            memoryMapping);
      }
      if ((allocFlags & PrefixSumAllocFlags::ALLOC_PREFIX_SUM) != 0) {
        buffers.prefixSum = alloc->createBuffer(Buffers::PrefixSumLayout::size(N),
            vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
      }
      buffers.decoupledStates = alloc->createBuffer(Buffers::DecoupledStatesLayout::size(partitionCount),
          vk::BufferUsageFlagBits::eTransferSrc,
          memoryMapping);
    }
    return buffers;
}
