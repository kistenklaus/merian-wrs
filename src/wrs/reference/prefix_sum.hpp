#pragma once
#include "src/wrs/why.hpp"
#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <vector>

namespace wrs::reference {

template <wrs::arithmetic T, wrs::typed_allocator<T> Allocator = std::allocator<T>>
std::vector<T, Allocator>
prefix_sum(const std::span<T> elements, bool ensureMonotone = true, const Allocator& alloc = {}) {
    std::vector<T, Allocator> prefix(elements.begin(), elements.end(), alloc);
    uint64_t N = elements.size();
    // Inital not work efficient algorithm:
    // - Reasonable numerical stability errors should not accumulate that much
    // - Problem: Reuslt is not guaranteed to be monotone when working with floating point numbers

    // Initialize Kahan summation variables
    double sum = 0.0f;
    double c = 0.0f; // compensation term

    for (uint64_t i = 0; i < N; ++i) {
        double y = prefix[i] - c; // subtract the previous compensation
        double t = sum + y;       // add the current element to the sum
        c = (t - sum) - y;       // calculate the new compensation
        sum = t;                 // update the sum

        prefix[i] = sum; // Store the current prefix sum
    }

    if (ensureMonotone) {
        // Fix monotone invariant!
        if constexpr (std::is_floating_point_v<T>) {
            // Perform a linear pass to check that the monotone invariant of prefix sum is not
            // broken!
            assert(prefix.size() == elements.size());
            for (size_t i = 1; i < prefix.size(); ++i) {
                T diff = prefix[i] - prefix[i - 1];
                if (elements[i] > 0) {
                    if (diff < 0) {
                        prefix[i] = prefix[i - 1];
                    }
                } else {
                    if (diff > 0) {
                        prefix[i] = prefix[i - 1];
                    }
                }
            }
        }
    }

    return prefix;
}

namespace pmr {

template <wrs::arithmetic T>
std::pmr::vector<T> prefix_sum(const std::span<T> weights,
                               bool ensureMonotone = true,
                               const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    // Hope for RTO
    return wrs::reference::prefix_sum<T, std::pmr::polymorphic_allocator<T>>(weights,
                                                                             ensureMonotone, alloc);
}

} // namespace pmr

} // namespace wrs::reference
