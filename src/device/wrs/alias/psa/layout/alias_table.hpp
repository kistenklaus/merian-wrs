#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "src/host/types/alias_table.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "src/host/layout/Attribute.hpp"
#include "src/host/layout/StructLayout.hpp"
#include "src/host/types/glsl.hpp"
#include <memory_resource>
namespace device::details {
namespace internals {

using AliasTableEntryLayout =
    host::layout::StructLayout<host::glsl::StorageQualifier::std430,
                               host::layout::Attribute<host::glsl::f32, "p">,
                               host::layout::Attribute<host::glsl::uint, "a">>;
}
using AliasTableLayout = host::layout::ArrayLayout<internals::AliasTableEntryLayout,
                                                   host::glsl::StorageQualifier::std430>;

merian::BufferHandle allocateAliasTableBuffer(const merian::ResourceAllocatorHandle& alloc,
                                              merian::MemoryMappingType memoryMapping,
                                              vk::BufferUsageFlags usageFlags,
                                              std::size_t N);

std::pmr::vector<host::AliasTableEntry<host::glsl::f32, host::glsl::uint>>
downloadAliasTableFromBuffer(
    const merian::BufferHandle& buffer,
    host::glsl::uint N,
    std::pmr::memory_resource* resource = std::pmr::get_default_resource());

} // namespace device::details
