#include "src/device/wrs/alias/psa_plus/layout/heavy_light_count.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include "vulkan/vulkan_handles.hpp"
#include <cstdlib>
#include <tuple>

std::tuple<host::glsl::uint, host::glsl::uint>
device::details::downloadHeavyLightCountFromStage(const merian::BufferHandle& buffer,
                                                  std::pmr::memory_resource* resource) {
    using View = host::layout::BufferView<device::details::HeavyLightCountLayout>;
    View view{buffer};
    auto heavyView = view.attribute<"heavyCount">();
    auto lightView = view.attribute<"lightCount">();
    return std::make_tuple(heavyView.download<host::glsl::uint>(),lightView.download<host::glsl::uint>());
}

merian::BufferHandle
device::details::allocateHeavyLightBuffer(const merian::ResourceAllocatorHandle& alloc,
                                          merian::MemoryMappingType memoryMapping,
                                          vk::BufferUsageFlags usageFlags) {
    return alloc->createBuffer(sizeof(host::glsl::uint) * 2, usageFlags, memoryMapping);
}
