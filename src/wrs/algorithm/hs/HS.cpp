#include "./HS.hpp"
#include <vulkan/vulkan_enums.hpp>

wrs::HSBuffers::Self wrs::HSBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                                              merian::MemoryMappingType memoryMapping,
                                              std::size_t N,
                                              std::size_t S,
                                              std::size_t explodePartitionSize) {

    std::size_t explodeWorkgroupCount = (N + explodePartitionSize - 1) / explodePartitionSize;

    hst::HSTRepr repr{N};
    std::size_t entries = repr.size();
    Self buffers;
    if (memoryMapping == merian::MemoryMappingType::NONE) {
        buffers.weightTree = alloc->createBuffer(WeightTreeLayout::size(entries),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferDst,
                                                 memoryMapping);
        buffers.outputSensitiveSamples =
            alloc->createBuffer(OutputSensitiveSamplesLayout::size(entries + 1),
                                vk::BufferUsageFlagBits::eStorageBuffer
                                | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.explodeDecoupledStates = alloc->createBuffer(
            ExplodeDecoupledStatesLayout::size(explodeWorkgroupCount),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            memoryMapping);
        buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferSrc,
                                              memoryMapping);

    } else {
        buffers.weightTree = alloc->createBuffer(
            WeightTreeLayout::size(entries), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.outputSensitiveSamples =
            alloc->createBuffer(OutputSensitiveSamplesLayout::size(entries + 1),
                                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        buffers.samples = alloc->createBuffer(SamplesLayout::size(S),
                                              vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
    }
    return buffers;
}
