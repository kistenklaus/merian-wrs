#pragma once

#include <algorithm>
#include <concepts>
#include <memory_resource>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>

namespace wrs::reference {

template <typename T, typename Allocator>
std::tuple<std::span<T>, std::span<T>, std::vector<T, Allocator>>
partition(const std::span<T> weights, T pivot, const Allocator& alloc = {}) {

    std::vector<T, Allocator> partition(std::ranges::begin(weights), std::ranges::end(weights),
                                        alloc);
    const auto mid = std::partition(partition.begin(), partition.end(),
                                    [pivot](const auto x) { return x > pivot; });
    return std::make_tuple(std::span<T>(partition.begin(), mid), std::span<T>(mid, partition.end()),
                           std::move(partition));
}

template <typename T, typename Allocator = std::allocator<T>>
std::tuple<std::span<T>, std::span<T>, std::vector<T, Allocator>>
stable_partition(const std::span<T> elements, T pivot, const Allocator& alloc = {}) {
    std::vector<T, Allocator> partition(elements.begin(), elements.end(), alloc);
    const auto mid = std::stable_partition(partition.begin(), partition.end(),
                                           [pivot](const T& x) { return x > pivot; });
    return std::make_tuple(std::span<T>(partition.begin(), mid), std::span<T>(mid, partition.end()),
                           std::move(partition));
}

namespace pmr {

template <typename T>
std::tuple<std::span<T>, std::span<T>, std::pmr::vector<T>>
partition(const std::span<T> weights, T pivot, const std::pmr::polymorphic_allocator<T>& alloc) {
    return std::move(
        wrs::reference::partition<T, std::pmr::polymorphic_allocator<T>>(weights, pivot, alloc));
}

template <typename T>
std::tuple<std::span<T>, std::span<T>, std::pmr::vector<T>>
stable_partition(const std::span<T> elements, T pivot, const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    return std::move(
        wrs::reference::stable_partition<T, std::pmr::polymorphic_allocator<T>>(elements, pivot, alloc));
}

} // namespace pmr

} // namespace wrs::reference
  //
  //
  //
  //
