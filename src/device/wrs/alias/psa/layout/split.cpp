#include "./split.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "src/host/layout/BufferView.hpp"

merian::BufferHandle
device::details::allocateSplitBuffer(const merian::ResourceAllocatorHandle& alloc,
                                    merian::MemoryMappingType memoryMapping,
                                    vk::BufferUsageFlags usageFlags,
                                    host::glsl::uint K) {

    // NOTE: Most LSPs can't handle this line therefor it's split into a seperate compliation unit.
    // Sorry if your LSP crashed try to reopen your editor and don't open this file again.
    return alloc->createBuffer(device::details::SplitsLayout::size(K), usageFlags, memoryMapping,
        "splits");
}


std::pmr::vector<host::Split<host::glsl::f32, host::glsl::uint>>
device::details::downloadSplitsFromBuffer(merian::BufferHandle buffer, std::size_t K,
    std::pmr::memory_resource* resource) {
  using View = host::layout::BufferView<SplitsLayout>;
  View view{buffer, K};
  // NOTE: Most LSPs can't handle this line therefor it's split into a seperate compliation unit.
  // Sorry if your LSP crashed try to reopen your editor and don't open this file again.
  return view.template download<host::Split<host::glsl::f32, host::glsl::uint>, host::pmr_alloc<host::Split<host::glsl::f32,
         host::glsl::uint>>>(resource); 
}

