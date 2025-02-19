#include "./DecoupledPartition.hpp"
#include "src/device/partition/PartitionAllocFlags.hpp"
#include "src/host/types/glsl.hpp"
#include <vulkan/vulkan_enums.hpp>


device::DecoupledPartitionBuffers::Self
device::DecoupledPartitionBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                         merian::MemoryMappingType memoryMapping,
                                         host::glsl::uint N,
                                         host::glsl::uint blockCount,
                                         PartitionAllocFlags allocFlags) {
    Self buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        if ((allocFlags & PartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
          buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferDst,
                                                 memoryMapping);
        }
        if ((allocFlags & PartitionAllocFlags::ALLOC_PIVOT) != 0) {
          buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
        }
        
        buffers.decoupledStates = alloc->createBuffer(DecoupledStatesLayout::size(blockCount),
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        
        if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
          buffers.partitionIndices = alloc->createBuffer(PartitionIndicesLayout::size(N),
                                                         vk::BufferUsageFlagBits::eStorageBuffer |
                                                             vk::BufferUsageFlagBits::eTransferSrc,
                                                         memoryMapping);
        }

        if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
          buffers.partition = alloc->createBuffer(PartitionLayout::size(N),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                  memoryMapping);
        }

        if ((allocFlags & PartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
          buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferSrc,
                                                   memoryMapping);
        }

    } else {

        if ((allocFlags & PartitionAllocFlags::ALLOC_ELEMENTS) != 0) {
          buffers.elements = alloc->createBuffer(
              ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }


        if ((allocFlags & PartitionAllocFlags::ALLOC_PIVOT) != 0) {
          buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        }

        buffers.decoupledStates = nullptr;

        if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_INDICES) != 0) {
          buffers.partitionIndices = alloc->createBuffer(
              PartitionIndicesLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        

        if ((allocFlags & PartitionAllocFlags::ALLOC_PARTITION_ELEMENTS) != 0) {
          buffers.partition = alloc->createBuffer(
              PartitionLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }


        if ((allocFlags & PartitionAllocFlags::ALLOC_HEAVY_COUNT) != 0) {
          buffers.heavyCount = alloc->createBuffer(
              HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
    }
    return buffers;
}

