#pragma once

#include "src/wrs/generic_types.hpp"
#include "src/wrs/why.hpp"
#include <fmt/base.h>
#include <memory>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace wrs::reference {

template <wrs::arithmetic T, std::integral I>
wrs::split_t<T, I>
split(const std::span<T> heavyPrefix, const std::span<T> lightPrefix, T mean, I n) {
    const I lightCount = lightPrefix.size();
    const I heavyCount = heavyPrefix.size();

    I a = 0;
    I b = std::min(n, heavyCount);

    T target = mean * n;

    while (true) {
        const I j = (a + b) / 2;                 // amount of heavy items
        const I i = std::min(n - j, lightCount); // amount of light items

        if (a > b) {
            const T light = lightPrefix[i];
            const T sigma2 = light + heavyPrefix[j + 1];
            return std::make_tuple(i, j, sigma2 - target);
        }
        const T heavy = j == 0 ? 0 : heavyPrefix[j - 1];
        const T light = i == 0 ? 0 : lightPrefix[i - 1];
        const T sigma = light + heavy;
        const T sigma2 = light + heavyPrefix[j];

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
          std::integral I,
          wrs::typed_allocator<wrs::split_t<T, I>> Allocator = std::allocator<wrs::split_t<T, I>>>
std::vector<wrs::split_t<T, I>, Allocator> splitK(const std::span<T> heavyPrefix,
                                                  const std::span<T> lightPrefix,
                                                  const T mean,
                                                  const I N,
                                                  const I K,
                                                  const Allocator& alloc = {}) {
    std::vector<wrs::split_t<T, I>, Allocator> splits(K, alloc);
    for (I k = 1; k <= K - 1; ++k) {
        const std::uintmax_t temp = static_cast<std::uintmax_t>(N) * static_cast<std::uintmax_t>(k);
        const I n =
            static_cast<I>(wrs::ceilDiv<std::uintmax_t>(temp, static_cast<std::uintmax_t>(K)));
        fmt::println("Computing split {}/{} with {} elements of {} elements", k, K, n, N);
        splits[k - 1] = split<T, I>(heavyPrefix, lightPrefix, mean, n);
    }
    splits.back() =
        std::make_tuple(static_cast<I>(lightPrefix.size()), static_cast<I>(heavyPrefix.size()), 0);
    return splits;
}

namespace pmr {

template <wrs::arithmetic T, std::integral I>
std::pmr::vector<wrs::split_t<T, I>> inline splitK(
    const std::span<T> heavyPrefix,
    const std::span<T> lightPrefix,
    const T mean,
    const I N,
    const I K,
    const std::pmr::polymorphic_allocator<wrs::split_t<T, I>>& alloc = {}) {
    // Hope for RTO
    return wrs::reference::splitK<T, I, std::pmr::polymorphic_allocator<wrs::split_t<T, I>>>(
        heavyPrefix, lightPrefix, mean, N, K, alloc);
}

} // namespace pmr

} // namespace wrs::reference
