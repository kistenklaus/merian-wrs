#include "./ScalarSplit.hpp"

wrs::ScalarSplitBuffers wrs::ScalarSplitBuffers::allocate(merian::ResourceAllocatorHandle alloc,
                                         std::size_t weightCount,
                                         std::size_t splitCount,
                                         merian::MemoryMappingType memoryMapping) {
        ScalarSplitBuffers buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.partitionPrefix =
                alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
                                    PARTITION_PREFIX_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.mean = alloc->createBuffer(MeanLayout::size(),
                MEAN_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            buffers.splits = alloc->createBuffer(SplitsLayout::size(splitCount),
                SPLITS_BUFFER_USAGE_FLAGS | vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
        } else {
            buffers.partitionPrefix =
                alloc->createBuffer(PartitionPrefixLayout::size(weightCount),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.mean = alloc->createBuffer(MeanLayout::size(),
                vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.splits = alloc->createBuffer(SplitsLayout::size(splitCount),
                vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }

