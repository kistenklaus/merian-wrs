#include "./Explode.hpp"

wrs::ExplodeBuffers::Self
wrs::ExplodeBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                              merian::MemoryMappingType memoryMapping,
                              std::size_t N,
                              std::size_t S,
                              std::size_t partitionSize) {
    Self buffers;
    std::size_t workgroupCount = (N + partitionSize - 1) / partitionSize;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.outputSensitive = alloc->createBuffer(OutputSensitiveLayout::size(N),
                                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                                          vk::BufferUsageFlagBits::eTransferDst,
                                                      memoryMapping);
        buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferSrc,
                                              memoryMapping);
        buffers.outputSensitive =
            alloc->createBuffer(DecoupledStatesLayout::size(workgroupCount), vk::BufferUsageFlagBits::eStorageBuffer, memoryMapping);
    } else {
        buffers.outputSensitive = alloc->createBuffer(
            OutputSensitiveLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                              vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        buffers.outputSensitive = nullptr;
    }
    return buffers;
}
