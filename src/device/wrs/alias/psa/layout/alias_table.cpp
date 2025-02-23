#include "./alias_table.hpp"
#include "src/host/layout/BufferView.hpp"

merian::BufferHandle
device::details::allocateAliasTableBuffer(const merian::ResourceAllocatorHandle& alloc,
                                          merian::MemoryMappingType memoryMapping,
                                          vk::BufferUsageFlags usageFlags,
                                          std::size_t N) {

    // NOTE: Most LSPs can't handle this line therefor it's split into a seperate compliation unit.
    // Sorry if your LSP crashed try to reopen your editor and don't open this file again.
    return alloc->createBuffer(device::details::AliasTableLayout::size(N),
        usageFlags, memoryMapping, "alias-table");
}

std::pmr::vector<host::AliasTableEntry<host::glsl::f32, host::glsl::uint>>
device::details::downloadAliasTableFromBuffer(const merian::BufferHandle& buffer,
                                              host::glsl::uint N,
                                              std::pmr::memory_resource* resource) {
  using View = host::layout::BufferView<AliasTableLayout>;
  View view{buffer, N};
  // NOTE: Most LSPs can't handle this line therefor it's split into a seperate compliation unit.
  // Sorry if your LSP crashed try to reopen your editor and don't open this file again.
  return view.template download<host::AliasTableEntry<host::glsl::f32, host::glsl::uint>, host::pmr_alloc<host::AliasTableEntry<host::glsl::f32,
         host::glsl::uint>>>(resource); 
}

