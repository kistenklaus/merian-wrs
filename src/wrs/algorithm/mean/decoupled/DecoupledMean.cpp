#include "./DecoupledMean.hpp"

wrs::DecoupledMeanBuffers wrs::DecoupledMeanBuffers::allocate(merian::ResourceAllocatorHandle alloc,
                                     std::size_t elementCount,
                                     std::size_t partitionSize,
                                     merian::MemoryMappingType memoryMapping) {
    std::size_t workgroupCount = (elementCount + partitionSize - 1) / partitionSize;
    DecoupledMeanBuffers buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.elements =
            alloc->createBuffer(ElementsLayout::size(elementCount),
                                ELEMENT_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(),
            MEAN_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.decoupledStates = alloc->createBuffer(DecoupledStatesLayout::size(workgroupCount),
            MEAN_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    } else {
        buffers.elements = alloc->createBuffer(ElementsLayout::size(elementCount),
                                               vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.mean = alloc->createBuffer(MeanLayout::size(),
            vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.decoupledStates = alloc->createBuffer(DecoupledStatesLayout::size(workgroupCount),
            {}, memoryMapping);
    }
    return buffers;
}
