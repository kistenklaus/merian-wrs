#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "src/host/layout/Attribute.hpp"
#include "src/host/layout/StructLayout.hpp"
#include "src/host/types/glsl.hpp"
#include "src/host/types/split.hpp"

#include <memory_resource>
#include <vector>

namespace device::details {

namespace internals {

using SplitStructLayout =
    host::layout::StructLayout<host::glsl::StorageQualifier::std430,
                               host::layout::Attribute<host::glsl::uint, "i">,
                               host::layout::Attribute<host::glsl::uint, "j">,
                               host::layout::Attribute<host::glsl::f32, "spill">>;

}

using SplitsLayout =
    host::layout::ArrayLayout<internals::SplitStructLayout, host::glsl::StorageQualifier::std430>;

merian::BufferHandle allocateSplitBuffer(const merian::ResourceAllocatorHandle& alloc,
                                         merian::MemoryMappingType memoryMapping,
                                         vk::BufferUsageFlags usageFlags,
                                         host::glsl::uint K);

std::pmr::vector<host::Split<host::glsl::f32, host::glsl::uint>>
downloadSplitsFromBuffer(merian::BufferHandle buffer, std::size_t K, std::pmr::memory_resource* memory_resource = 
    std::pmr::get_default_resource());

} // namespace device::details
