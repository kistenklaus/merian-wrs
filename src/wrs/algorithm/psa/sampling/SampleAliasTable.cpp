#include "./SampleAliasTable.hpp"

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

wrs::SampleAliasTableBuffers wrs::SampleAliasTableBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t weightCount,
                         std::size_t sampleCount) {
        using Self = wrs::SampleAliasTableBuffers;
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
           buffers.aliasTable = alloc->createBuffer(Self::AliasTableLayout::size(weightCount),  // bye bye LSP!
               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
               merian::MemoryMappingType::NONE); 
          buffers.samples = alloc->createBuffer(Self::SamplesLayout::size(sampleCount),
              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
              merian::MemoryMappingType::NONE);
        } else { 
           buffers.aliasTable = alloc->createBuffer(Self::AliasTableLayout::size(weightCount),  // bye bye LSP!
               vk::BufferUsageFlagBits::eTransferSrc, 
               memoryMapping); 
          buffers.samples = alloc->createBuffer(Self::SamplesLayout::size(sampleCount),
              vk::BufferUsageFlagBits::eTransferDst,
              memoryMapping);
        }
        return buffers;
}

