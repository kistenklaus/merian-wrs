#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "src/host/layout/Attribute.hpp"
#include "src/host/layout/StructLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <memory_resource>

namespace device::details {

using HeavyLightCountLayout =
    host::layout::StructLayout<host::glsl::StorageQualifier::std430,
                               host::layout::Attribute<host::glsl::uint, "heavyCount">,
                               host::layout::Attribute<host::glsl::uint, "lightCount">>;

merian::BufferHandle allocateHeavyLightBuffer(const merian::ResourceAllocatorHandle& alloc,
                                              merian::MemoryMappingType memoryMapping,
                                              vk::BufferUsageFlags usageFlags);

std::tuple<host::glsl::uint, host::glsl::uint>
downloadHeavyLightCountFromStage(
    const merian::BufferHandle& buffer,
    std::pmr::memory_resource* resource = std::pmr::get_default_resource());
}
