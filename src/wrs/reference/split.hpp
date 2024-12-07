#pragma once

#include "src/wrs/generic_types.hpp"
#include "src/wrs/why.hpp"
#include <fmt/base.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace wrs::reference {

template <wrs::arithmetic T>
wrs::split_t<T>
split(const std::span<T> heavyPrefix, const std::span<T> lightPrefix, T mean, size_t n) {
    size_t a = 0;
    size_t b = std::min(n, heavyPrefix.size()) - 1;

    T target = mean * n;

    while (true) {
        const size_t j = (a + b) / 2;
        const size_t i = std::min(n - j, lightPrefix.size() - 1);

        if (a > b) {
            const T light = lightPrefix[i];
            const T sigma2 = light + heavyPrefix[j + 1];
            return std::make_tuple(i, j, sigma2 - target);
        }

        const T heavy = heavyPrefix[j];
        const T light = lightPrefix[i];
        const T sigma = light + heavy;
        const T sigma2 = light + heavyPrefix[j + 1];

        if (sigma <= target) {
            if (sigma2 > target) {
                return std::make_tuple(i, j, sigma2 - target);
            } else {
                a = j + 1;
            }
        } else {
            b = j - 1;
        }
    }
    throw std::runtime_error("F");
}

template <wrs::arithmetic T,
          wrs::typed_allocator<wrs::split_t<T>> Allocator = std::allocator<wrs::split_t<T>>>
std::vector<wrs::split_t<T>, Allocator> splitK(const std::span<T> heavyPrefix,
                                               const std::span<T> lightPrefix,
                                               T mean,
                                               size_t N,
                                               size_t K,
                                               const Allocator& alloc = {}) {
    std::vector<wrs::split_t<T>, Allocator> splits(K, alloc);
    for (size_t k = 1; k <= K - 1; ++k) {
        const size_t temp = N * k;
        const size_t n = 1 + ((temp - 1) / K);
        splits[k - 1] = split(heavyPrefix, lightPrefix, mean, n);
    }
    splits.back() = std::make_tuple(N, N, 0);
    return splits;
}

namespace pmr {

template <wrs::arithmetic T>
std::pmr::vector<wrs::split_t<T>> inline splitK(
    const std::span<T> heavyPrefix,
    const std::span<T> lightPrefix,
    T mean,
    size_t N,
    size_t K,
    const std::pmr::polymorphic_allocator<wrs::split_t<T>>& alloc = {}) {
    // Hope for RTO
    return wrs::reference::splitK<T, std::pmr::polymorphic_allocator<wrs::split_t<T>>>(
        heavyPrefix, lightPrefix, mean, N, K, alloc);
}

} // namespace pmr

} // namespace wrs::reference
