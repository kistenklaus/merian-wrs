
#include "./DecoupledPrefixPartition.hpp"

wrs::DecoupledPrefixPartitionBuffers
wrs::DecoupledPrefixPartitionBuffers::allocate(merian::ResourceAllocatorHandle alloc,
                                        std::size_t elementCount,
                                        std::size_t partitionSize,
                                        merian::MemoryMappingType memoryMapping) {
    std::size_t workgroupCount = (elementCount + partitionSize - 1) / partitionSize;
    DecoupledPrefixPartitionBuffers buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.elements = alloc->createBuffer(
            ElementsLayout::size(elementCount),
            ELEMENT_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.pivot = alloc->createBuffer(
            PivotLayout::size(), PIVOT_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        buffers.batchDescriptors = alloc->createBuffer(BatchDescriptorsLayout::size(workgroupCount),
                                                       BATCH_DESCRIPTOR_BUFFER_USAGE_FLAGS |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       memoryMapping);
        buffers.partitionPrefix = alloc->createBuffer(
            PartitionPrefixLayout::size(elementCount),
            PREFIX_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.partition = alloc->createBuffer(
            PartitionLayout::size(elementCount),
            PARTITION_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
    } else {
        buffers.elements =
            alloc->createBuffer(ElementsLayout::size(elementCount),
                                vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.pivot = alloc->createBuffer(PivotLayout::size(),
                                            vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.batchDescriptors = alloc->createBuffer(BatchDescriptorsLayout::size(workgroupCount),
                                                       vk::BufferUsageFlagBits::eTransferSrc |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       memoryMapping);
        buffers.partitionPrefix =
            alloc->createBuffer(PartitionPrefixLayout::size(elementCount),
                                vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.partition =
            alloc->createBuffer(PartitionLayout::size(elementCount),
                                vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    }
    return buffers;
}
