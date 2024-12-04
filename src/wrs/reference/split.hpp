#pragma once

#include <fmt/base.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace wrs::reference {

namespace internal {
template <typename T> using Split = std::tuple<std::size_t, std::size_t, T>;

// Save index operator for spans. Only available stl in C++26 (for some reason??).
template <typename T> T& spanAt(std::span<T> span, size_t i) {
    if (i < span.size()) {
        return span[i];
    } else {
        throw std::out_of_range("Accesing span out of range");
    }
}

} // namespace internal

template <typename T>
internal::Split<T>
split(const std::span<T> heavyPrefix, const std::span<T> lightPrefix, T mean, size_t n) {
    size_t a = 0;
    size_t b = std::min(n, heavyPrefix.size()) - 1;

    T target = mean * n;

    while (true) {
        const size_t j = (a + b) / 2;
        const size_t i = std::min(n - j, lightPrefix.size() - 1);


        const T heavy = internal::spanAt(heavyPrefix, j);
        const T light = internal::spanAt(lightPrefix, i);
        const T sigma = light + heavy;
        const T sigma2 = light + internal::spanAt(heavyPrefix, j + 1);

        if (a > b) {
            return std::make_tuple(i, j, sigma2 - target);
        }

        if (sigma <= target && sigma2 > target) {
            return std::make_tuple(i, j, sigma2 - target);
        } else if (sigma <= target) {
            a = j + 1;
        } else {
            b = j - 1;
        }
    }
    throw std::runtime_error("F");
}

template <typename T, typename Allocator = std::allocator<internal::Split<T>>>
std::vector<internal::Split<T>, Allocator> splitK(const std::span<T> heavyPrefix,
                                                  const std::span<T> lightPrefix,
                                                  T mean,
                                                  size_t N,
                                                  size_t K,
                                                  const Allocator& alloc = {}) {
    std::vector<internal::Split<T>, Allocator> splits(K, alloc);
    for (size_t k = 1; k <= K - 1; ++k) {
        const size_t temp = N * k;
        const size_t n = 1 + ((temp - 1) / K);
        splits[k - 1] = split(heavyPrefix, lightPrefix, mean, n);
    }
    splits.back() = std::make_tuple(N, N, 0);
    return splits;
}

namespace pmr {

template <typename T>
std::pmr::vector<internal::Split<T>>
splitK(const std::span<T> heavyPrefix,
       const std::span<T> lightPrefix,
       T mean,
       size_t N,
       size_t K,
       const std::pmr::polymorphic_allocator<internal::Split<T>>& alloc = {}) {
    // Now that is a signiture =^).
    return std::move(wrs::reference::splitK<T, std::pmr::polymorphic_allocator<internal::Split<T>>>(
        heavyPrefix, lightPrefix, mean, N, K, alloc));
}

} // namespace pmr

} // namespace wrs::reference
