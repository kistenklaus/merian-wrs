#pragma once

#include "src/wrs/why.hpp"
#include <algorithm>
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

template <typename T>
T neumaier_reduction(std::span<const T> elements) {
    T sum = 0;
    T c = 0; // Error term
    
    for (const T& element : elements) {
        T y = element - c;
        T t = sum + y;
        c = (t - sum) - y;  // Update error term
        sum = t;
    }
    
    return sum;
}

template <typename T>
T pairwise_kahan_reduction(const std::span<T> elements) {
    std::vector<T> summed_elements{elements.begin(),elements.end()};

    // Pairwise summation
    while (summed_elements.size() > 1) {
        std::vector<T> next_summed_elements;
        for (size_t i = 0; i < summed_elements.size(); i += 2) {
            if (i + 1 < summed_elements.size()) {
                T a = summed_elements[i];
                T b = summed_elements[i + 1];
                T sum = a + b;
                T c = (sum - a) - b;
                next_summed_elements.push_back(sum - c);
            } else {
                next_summed_elements.push_back(summed_elements[i]);
            }
        }
        summed_elements = next_summed_elements;
    }

    return summed_elements[0];
}

template <typename T>
T kahan_reduction(std::span<const T> elements) {
    T sum = 0;          // Running total sum
    T c = 0;            // Compensation term
    for (const T& element : elements) {
        T y = element - c;     // Subtract compensation from current element
        T t = sum + y;         // Add the adjusted value to the sum
        c = (t - sum) - y;     // Calculate the new compensation
        sum = t;               // Update the running total
    }

    return sum;
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
