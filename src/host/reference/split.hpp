#pragma once

#include "src/host/types/split.hpp"
#include "src/host/why.hpp"
#include <fmt/base.h>
#include <memory>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace host::reference {

template <arithmetic T, std::integral I>
Split<T, I> split(std::span<const T> heavyPrefix, std::span<const T> lightPrefix, T mean, I n) {
    const I lightCount = lightPrefix.size();
    const I heavyCount = heavyPrefix.size();

    I a = 0;
    I b = std::min(n, heavyCount);

    T target = mean * n;

    while (true) {
        const I j = (a + b) / 2;                 // amount of heavy items
        const I i = std::min(n - j, lightCount); // amount of light items

        if (a > b) {
            assert(i < lightCount);
            const T light = lightPrefix[i];
            assert(j + 1 < heavyCount);
            const T sigma2 = light + heavyPrefix[j + 1];
            return Split(i, j, sigma2 - target);
        }
        assert((j == 0) || (j - 1 < heavyCount));
        const T heavy = j == 0 ? 0 : heavyPrefix[j - 1];
        assert((i == 0) || (i - 1 < lightCount));
        const T light = i == 0 ? 0 : lightPrefix[i - 1];
        const T sigma = light + heavy;
        if (j == heavyCount) {
            fmt::println("sigma = {}, target = {}", sigma, target);
        }
        assert(j != heavyCount);
        const T sigma2 = light + heavyPrefix[j];

        if (sigma <= target) {
            if (sigma2 > target) {
                return Split(i, j, sigma2 - target);
            } else {
                a = j + 1;
            }
        } else {
            b = j - 1;
        }
    }
    throw std::runtime_error("F");
}

template <std::integral I> I ceilMulDiv(I a, I b, I c) {
    I div = a / c;
    I rem = a % c;
    I safe_part = div * b;
    I rem_part = (rem * b + c - 1) / c;
    return safe_part + rem_part;
}

template <arithmetic T,
          std::integral I,
          typed_allocator<Split<T, I>> Allocator = std::allocator<Split<T, I>>>
std::vector<Split<T, I>, Allocator> splitK(std::span<const T> heavyPrefix,
                                           std::span<const T> lightPrefix,
                                           const T mean,
                                           const I N,
                                           const I K,
                                           const Allocator& alloc = {}) {
    std::vector<Split<T, I>, Allocator> splits(K, alloc);
    for (I k = 1; k <= K - 1; ++k) {
        const I n = ceilMulDiv(N, static_cast<I>(k), static_cast<I>(K));
        ;
        /* fmt::println("Computing split {}/{} with {} elements of {} elements", k, K, n, N); */
        splits[k - 1] = split<T, I>(heavyPrefix, lightPrefix, mean, n);
    }
    splits.back() =
        Split(static_cast<I>(lightPrefix.size()), static_cast<I>(heavyPrefix.size() - 1), T{0});
    return splits;
}

namespace pmr {
template <arithmetic T, std::integral I>
std::pmr::vector<Split<T, I>> inline splitK(
    std::span<const T> heavyPrefix,
    std::span<const T> lightPrefix,
    const T mean,
    const I N,
    const I K,
    const std::pmr::polymorphic_allocator<Split<T, I>>& alloc = {}) {
    // Hope for RTO
    return reference::splitK<T, I, std::pmr::polymorphic_allocator<Split<T, I>>>(
        heavyPrefix, lightPrefix, mean, N, K, alloc);
}
} // namespace pmr
} // namespace host::reference
