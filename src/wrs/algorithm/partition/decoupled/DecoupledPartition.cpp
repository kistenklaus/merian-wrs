#include "./DecoupledPartition.hpp"
#include <vulkan/vulkan_enums.hpp>


wrs::DecoupledPartitionBuffers::Self
wrs::DecoupledPartitionBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                         merian::MemoryMappingType memoryMapping,
                                         glsl::uint N,
                                         glsl::uint blockCount) {
    Self buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               memoryMapping);
        buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                            vk::BufferUsageFlagBits::eStorageBuffer |
                                                vk::BufferUsageFlagBits::eTransferDst,
                                            memoryMapping);

        buffers.decoupledStates = alloc->createBuffer(DecoupledStatesLayout::size(blockCount),
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        

        buffers.partitionIndices = alloc->createBuffer(PartitionIndicesLayout::size(N),
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferSrc,
                                                       memoryMapping);

        buffers.partition = alloc->createBuffer(PartitionLayout::size(N),
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferSrc,
                                                memoryMapping);

        buffers.heavyCount = alloc->createBuffer(HeavyCountLayout::size(),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferSrc,
                                                 memoryMapping);

    } else {
        buffers.elements = alloc->createBuffer(
            ElementsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);

        buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                            vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);

        buffers.decoupledStates = nullptr;

        buffers.partitionIndices = alloc->createBuffer(
            PartitionIndicesLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

        buffers.partition = alloc->createBuffer(
            PartitionLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);

        buffers.heavyCount = alloc->createBuffer(
            HeavyCountLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    }
    return buffers;
}

