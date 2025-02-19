#pragma once

#include "src/host/types/partition.hpp"
#include "src/host/why.hpp"
#include <algorithm>
#include <cassert>
#include <fmt/base.h>
#include <limits>
#include <memory_resource>
#include <ranges>
#include <span>
#include <vector>

namespace host::reference {

template <arithmetic T, typed_allocator<T> Allocator = std::allocator<T>>
Partition<T, std::vector<T, Allocator>>
partition(std::span<const T> elements, T pivot, const Allocator& alloc = {}) {
    std::vector<T, Allocator> partition(std::ranges::begin(elements), std::ranges::end(elements),
                                        alloc);
    const auto mid = std::partition(partition.begin(), partition.end(),
                                    [pivot](const auto x) { return x > pivot; });
    return Partition<T, std::vector<T, Allocator>>(std::move(partition), mid - partition.begin());
}

template <arithmetic T, typed_allocator<T> Allocator = std::allocator<T>>
Partition<T, std::vector<T, Allocator>>
stable_partition(std::span<const T> elements, T pivot, const Allocator& alloc = {}) {
    std::vector<T, Allocator> partition(elements.begin(), elements.end(), alloc);
    const auto mid = std::stable_partition(partition.begin(), partition.end(),
                                           [pivot](const T& x) { return x > pivot; });
    return Partition<T, std::vector<T, Allocator>>(std::move(partition), mid - partition.begin());
}

template <arithmetic T, std::integral I, typed_allocator<I> Allocator = std::allocator<I>>
Partition<I, std::vector<I, Allocator>>
stable_partition_indicies(std::span<const T> elements, const T pivot, const Allocator& alloc = {}) {
    assert(std::numeric_limits<I>::max() > elements.size());
    std::vector<I, Allocator> indices{elements.size(), alloc};
    const I N = static_cast<I>(elements.size());
    I l = N - 1;
    I h = 0;
    for (I i = 0; i < N; ++i) {
        const T& w = elements[i];
        if (w > pivot) {
            indices[h++] = i;
        } else {
            indices[l--] = i;
        }
    }
    std::reverse(indices.begin() + h, indices.end());
    return Partition<I, std::vector<I, Allocator>>(std::move(indices), h);
}

namespace pmr {

template <arithmetic T>
inline Partition<T, std::pmr::vector<T>>
partition(const std::span<T> elements, T pivot, const std::pmr::polymorphic_allocator<T>& alloc) {
    // URTO
    return reference::partition<T, std::pmr::polymorphic_allocator<T>>(elements, pivot, alloc);
}

template <arithmetic T>
inline Partition<T, std::pmr::vector<T>> stable_partition(
    const std::span<T> elements, T pivot, const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    return reference::stable_partition<T, std::pmr::polymorphic_allocator<T>>(elements, pivot,
                                                                              alloc);
}

template <arithmetic T, std::integral I>
inline Partition<I, std::pmr::vector<I>>
stable_partition_indicies(std::span<const T> elements,
                          const T pivot,
                          const std::pmr::polymorphic_allocator<I>& alloc = {}) {
    // URTO
    return reference::stable_partition_indicies<T, I, std::pmr::polymorphic_allocator<I>>(
        elements, pivot, alloc);
}

} // namespace pmr

} // namespace host::reference
