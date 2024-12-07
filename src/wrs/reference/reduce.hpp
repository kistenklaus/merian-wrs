#pragma once

#include "src/wrs/why.hpp"
#include <cassert>
#include <concepts>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <memory>
#include <numeric>
#include <span>
#include <vector>
namespace wrs::reference {

/**
 * Performs a reduction operation with the binary + operator!
 * Uses a reasonably numerically stable algorithm!
 *
 * We have implemented a hierarchical reduction algorithm, which
 * guarantees that calls to the binary + operator is called only
 * partial reductions of around the same amount of elements.
 * Compared to a linear scan reduction this reduces numerical
 * instabilities for large input sizes.
 * (e.g avoids a + b = a if a >>> 0 and 0 < b ~ 0).
 *
 * Current implementation is unaware the the values of elements,
 * future implementation could utilize this to improve stability.
 * (e.g inital sorting).
 */
template <wrs::arithmetic T, wrs::typed_allocator<T> Allocator = std::allocator<T>>
    requires(std::same_as<typename Allocator::value_type, T>)
T tree_reduction(const std::span<T> elements, const Allocator& alloc = {}) {
    std::vector<T, Allocator> scratch(elements.begin(), elements.end(), alloc);
    for (size_t split = std::bit_ceil(elements.size()) >> 1; split > 0; split >>= 1) {
        for (size_t j = 0; j < split; j++) {
            if (j + split < scratch.size()) {
                scratch[j] = scratch[j] + scratch[j + split];
            }
        }
    }
    return scratch.front();
}

template <wrs::arithmetic T, wrs::typed_allocator<T> Allocator = std::allocator<T>>
    requires(std::same_as<typename Allocator::value_type, T>)
T block_reduction(const std::span<T> elements, std::size_t blockSize, const Allocator& alloc = {}) {
    assert(blockSize > 1);
    std::span<T> elements2 = elements;
    std::size_t N = elements.size();
    std::size_t numBlocks = (N + blockSize - 1) / blockSize;
    std::vector<T, Allocator> blockSums{numBlocks, alloc};
    while (N > blockSize) {
        // unsigned underflow!!!
        for (std::size_t block = numBlocks - 1; block < numBlocks; --block) {
            /* fmt::println("block {}", block); */
            std::size_t blockOffset = block * blockSize;
            /* fmt::println("Acc"); */
            blockSums[block] =
                std::accumulate(elements2.begin() + blockOffset,
                                elements2.begin() + std::min(blockOffset + blockSize, N), T{});
            /* fmt::println("Acc Done"); */
        }
        /* fmt::println("OUT OF LOOP = {:?}", blockSums); */
        N = numBlocks;
        numBlocks = (N + blockSize - 1) / blockSize;
        elements2 = blockSums;
    }
    assert(N <= blockSize);
    return std::accumulate(blockSums.begin(), blockSums.begin() + N, T{});
}

namespace pmr {

/**
 * Performs a reduction operation with the binary + operator!
 * Uses a reasonably numerically stable algorithm!
 *
 * We have implemented a hierarchical reduction algorithm, which
 * guarantees that calls to the binary + operator is called only
 * partial reductions of around the same amount of elements.
 * Compared to a linear scan reduction this reduces numerical
 * instabilities for large input sizes.
 * (e.g avoids a + b = a if a >>> 0 and 0 < b ~ 0).
 *
 * Current implementation is unaware the the values of elements,
 * future implementation could utilize this to improve stability.
 * (e.g inital sorting).
 */
template <wrs::arithmetic T>
inline T tree_reduction(const std::span<T> elements,
                        const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    return wrs::reference::tree_reduction(elements, alloc);
}

template <wrs::arithmetic T>
inline T block_reduction(const std::span<T> elements,
                         std::size_t blockSize,
                         const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    return wrs::reference::block_reduction<T>(elements, blockSize, alloc);
}

} // namespace pmr

} // namespace wrs::reference
