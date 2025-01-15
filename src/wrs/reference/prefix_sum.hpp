#pragma once
#include "src/wrs/why.hpp"
#include <concepts>
#include <cstdint>
#include <memory_resource>
#include <random>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

namespace wrs::reference {

template <arithmetic T,
          typed_allocator<T> Allocator = std::allocator<T>,
          std::ranges::random_access_range Range = std::span<const T>>
    requires(std::convertible_to<std::ranges::range_value_t<Range>, T>)
std::vector<T, Allocator> prefix_sum(const Range& elements, const Allocator& alloc = {}) {
    std::vector<T, Allocator> prefix(elements.begin(), elements.end(), alloc);
    const uint64_t N = elements.size();
    // Inital not work efficient algorithm:
    // - Reasonable numerical stability errors should not accumulate that much
    // - Problem: Result is not guaranteed to be monotone when working with floating point numbers

    // Initialize Kahan summation variables
    T sum = 0.0f;
    T c = 0.0f; // compensation term

    for (uint64_t i = 0; i < N; ++i) {
        T y = prefix[i] - c; // subtract the previous compensation
        T t = sum + y;       // add the current element to the sum
        c = (t - sum) - y;   // calculate the new compensation
        sum = t;             // update the sum

        prefix[i] = sum; // Store the current prefix sum
    }
    return prefix;
}

template <arithmetic T,
          typed_allocator<T> Allocator = std::allocator<T>,
          std::ranges::random_access_range Range = std::span<const T>>
    requires(std::convertible_to<std::ranges::range_value_t<Range>, T>)
std::vector<T, Allocator>
imperfect_prefix_sum(const Range& elements, T std_deviation, const Allocator& alloc = {}) {
    auto prefixSum = prefix_sum<T, Allocator, Range>(elements, alloc);
    if (std_deviation > 0) {
        std::normal_distribution<T> dist{0, std_deviation};
        std::mt19937 rng{1234};
        for (auto& e : prefixSum) {
            e += dist(rng);
        }
    }
    return prefixSum;
}

namespace pmr {

template <arithmetic T, std::ranges::random_access_range Range>
    requires(std::convertible_to<std::ranges::range_value_t<Range>, T>)
std::pmr::vector<T> prefix_sum(const Range& weights,
                               const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    // Hope for RTO
    return wrs::reference::prefix_sum<T, std::pmr::polymorphic_allocator<T>>(weights, alloc);
}

template <arithmetic T, std::ranges::random_access_range Range = std::span<const T>>
    requires(std::convertible_to<std::ranges::range_value_t<Range>, T>)
std::pmr::vector<T> imperfect_prefix_sum(const Range& elements,
                                         T std_deviation,
                                         const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    return wrs::reference::imperfect_prefix_sum<T, std::pmr::polymorphic_allocator<T>, Range>(
        elements, std_deviation, alloc);
}

} // namespace pmr

} // namespace wrs::reference
