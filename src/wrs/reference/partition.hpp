#pragma once

#include "src/wrs/generic_types.hpp"
#include "src/wrs/why.hpp"
#include <algorithm>
#include <memory_resource>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>

namespace wrs::reference {

template <wrs::arithmetic T, wrs::typed_allocator<T> Allocator = std::allocator<T>>
wrs::partition_t<T, Allocator>
partition(const std::span<T> elements, T pivot, const Allocator& alloc = {}) {

    std::vector<T, Allocator> partition(std::ranges::begin(elements), std::ranges::end(elements),
                                        alloc);
    const auto mid = std::partition(partition.begin(), partition.end(),
                                    [pivot](const auto x) { return x > pivot; });
    return std::make_tuple(std::span<T>(partition.begin(), mid), std::span<T>(mid, partition.end()),
                           std::move(partition));
}

template <wrs::arithmetic T, wrs::typed_allocator<T> Allocator = std::allocator<T>>
wrs::partition_t<T, Allocator>
stable_partition(const std::span<T> elements, T pivot, const Allocator& alloc = {}) {
    std::vector<T, Allocator> partition(elements.begin(), elements.end(), alloc);
    const auto mid = std::stable_partition(partition.begin(), partition.end(),
                                           [pivot](const T& x) { return x > pivot; });
    return std::make_tuple(std::span<T>(partition.begin(), mid), std::span<T>(mid, partition.end()),
                           std::move(partition));
}

template <wrs::arithmetic T,
          std::integral I,
          wrs::typed_allocator<std::size_t> Allocator = std::allocator<std::size_t>>
wrs::partition_indices_t<Allocator, I> stable_partition_indicies(const std::span<T>& elements,
                                                                 const T pivot,
                                                                 const Allocator& alloc = {}) {}

namespace pmr {

template <wrs::arithmetic T>
inline wrs::pmr::partition_t<T>
partition(const std::span<T> elements, T pivot, const std::pmr::polymorphic_allocator<T>& alloc) {
    // Hope for RTO
    return wrs::reference::partition<T, std::pmr::polymorphic_allocator<T>>(elements, pivot, alloc);
}

template <wrs::arithmetic T>
inline wrs::pmr::partition_t<T> stable_partition(
    const std::span<T> elements, T pivot, const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    // Hope for RTO
    return wrs::reference::stable_partition<T, std::pmr::polymorphic_allocator<T>>(elements, pivot,
                                                                                   alloc);
}

} // namespace pmr

} // namespace wrs::reference
  //
  //
  //
  //
