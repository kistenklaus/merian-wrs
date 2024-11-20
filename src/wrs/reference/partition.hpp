#pragma once

#include <algorithm>
#include <concepts>
#include <memory_resource>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>
namespace wrs::reference {

template <typename T, typename ForwardRange, typename Allocator>
    requires std::same_as<T, std::ranges::range_value_t<ForwardRange>> &&
             std::ranges::forward_range<ForwardRange>
std::tuple<std::span<T>, std::span<T>, std::vector<T, Allocator>>
partition(const ForwardRange& weights, T pivot, const Allocator& alloc) {

    std::vector<T, Allocator> partition(std::ranges::begin(weights), std::ranges::end(weights),
                                        alloc);
    const auto mid = std::partition(partition.begin(), partition.end(),
                                    [pivot](const auto x) { return x > pivot; });
    return std::make_tuple(std::span<T>(partition.begin(), mid),
                           std::span<T>(mid, partition.end()), std::move(partition));
}

namespace pmr {

template <typename T, typename ForwardRange>
    requires std::same_as<T, std::ranges::range_value_t<ForwardRange>> &&
             std::ranges::forward_range<ForwardRange>
std::tuple<std::span<T>, std::span<T>, std::pmr::vector<T>>
partition(const ForwardRange& weights, T pivot, const std::pmr::polymorphic_allocator<T>& alloc) {
    return std::move(wrs::reference::partition<T, ForwardRange, std::pmr::polymorphic_allocator<T>>(
        weights, pivot, alloc));
}

} // namespace pmr

} // namespace wrs::reference
