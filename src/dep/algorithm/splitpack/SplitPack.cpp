#include "./SplitPack.hpp"
#include <vulkan/vulkan_enums.hpp>

wrs::SplitPackBuffers::Self
wrs::SplitPackBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                merian::MemoryMappingType memoryMapping,
                                glsl::uint N) {
    Self buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              memoryMapping);
        buffers.partitionIndices = alloc->createBuffer(
            PartitionIndicesLayout::size(N), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        buffers.partitionPrefix = alloc->createBuffer(
            PartitionPrefixLayout::size(N), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        buffers.mean = alloc->createBuffer(
            MeanLayout::size(), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        buffers.aliasTable = alloc->createBuffer(
            AliasTableLayout::size(N), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            memoryMapping);
        buffers.splits = alloc->createBuffer(
            SplitsLayout::size(N), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            memoryMapping);

    } else {
        buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                              vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.partitionIndices =
            alloc->createBuffer(PartitionIndicesLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.partitionPrefix =
            alloc->createBuffer(PartitionPrefixLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.aliasTable =
            alloc->createBuffer(AliasTableLayout::size(N), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.splits = alloc->createBuffer(
            SplitsLayout::size(N), vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
    }
    return buffers;
}
