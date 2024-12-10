#pragma once

#include "src/wrs/generic_types.hpp"
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
namespace wrs::reference {

template <wrs::arithmetic T, std::floating_point P, std::integral I,wrs::generic_allocator Allocator = std::allocator<void>>
std::vector<
    wrs::alias_table_entry_t<P, I>,
    typename std::allocator_traits<Allocator>::template rebind_alloc<wrs::alias_table_entry_t<P, I>>>
psa_alias_table(const std::span<T>& weights, const T totalWeight, const Allocator& alloc = {}) {

    using Entry = wrs::alias_table_entry_t<P, I>;
    using EntryAllocator = std::allocator_traits<Allocator>::template rebind_alloc<Entry>;

    std::size_t N = weights.size();
    const P averageWeight = static_cast<P>(totalWeight) / static_cast<P>(N);

    std::vector<Entry, EntryAllocator> aliasTable{N, EntryAllocator{alloc}};

    throw std::runtime_error("NOT IMPLEMENTED YET");
}

namespace pmr {
template <wrs::arithmetic T, std::floating_point P, std::integral I>
std::vector<wrs::alias_table_entry_t<P, I>, std::pmr::polymorphic_allocator<wrs::alias_table_entry_t<P, I>>>
psa_alias_table(const std::span<T>& weights,
                const T totalWeight,
                const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::reference::psa_alias_table<T, P, I, std::pmr::polymorphic_allocator<void>>(
        weights, totalWeight, alloc);
}

} // namespace pmr

} // namespace wrs::reference
