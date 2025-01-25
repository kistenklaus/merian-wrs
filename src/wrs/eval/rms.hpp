#pragma once

#include "src/wrs/eval/histogram.hpp"
#include "src/wrs/reference/reduce.hpp"
#include <cmath>
#include <concepts>
#include <fmt/base.h>
#include <glm/ext/scalar_constants.hpp>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <spdlog/spdlog.h>
namespace wrs::eval {

template <std::floating_point E,
          std::integral I,
          wrs::typed_allocator<I> Allocator = std::allocator<I>>
E rmse(std::span<const E> weights,
       std::span<const I> samples,
       std::optional<E> totalWeight = std::nullopt,
       const Allocator& alloc = {}) {
    assert(!weights.empty());
    assert(!samples.empty());
    if (!totalWeight.has_value()) {
        totalWeight = wrs::reference::kahan_reduction<E>(weights);
    }
    assert(totalWeight.has_value());

    if (*totalWeight < std::numeric_limits<E>::epsilon()) {
        SPDLOG_WARN(fmt::format(
            "Computing rmse_curve with a very small total weight might lead to invalid results.\n",
            "totalWeight = {}", *totalWeight));
    }

    std::vector<I, Allocator> histogram = wrs::eval::histogram(samples, weights.size(), alloc);

    std::size_t n = weights.size();
    E c = E{};
    E rmse{};
    for (size_t i = 0; i < n; ++i) {
        E w = weights[i];
        assert(w >= 0);
        const E expected = weights[i] / *totalWeight;
        const E observed = static_cast<E>(histogram[i]) / samples.size();
        const E diff = expected - observed;
        /* const E diff = (w * n - histogram[i] * totalWeight.value()) / (*totalWeight * n); */
        const E diff2 = diff * diff;
        const E term = diff2 - c;
        const E temp = rmse + term;
        c = (temp - rmse) - term;
        rmse = temp;
    }
    rmse = sqrt(rmse / static_cast<E>(weights.size()));
    return rmse;
}

template <std::floating_point E,
          std::integral I,
          wrs::typed_allocator<I> Allocator = std::allocator<I>>
E histogram_rmse(std::span<const E> weights,
       std::span<const I> histogram,
       std::optional<E> totalWeight = std::nullopt,
       const Allocator& alloc = {}) {
    assert(!weights.empty());
    assert(!histogram.empty());
    if (!totalWeight.has_value()) {
        totalWeight = wrs::reference::kahan_reduction<E>(weights);
    }
    assert(totalWeight.has_value());

    if (*totalWeight < std::numeric_limits<E>::epsilon()) {
        SPDLOG_WARN(fmt::format(
            "Computing rmse_curve with a very small total weight might lead to invalid results.\n",
            "totalWeight = {}", *totalWeight));
    }

    std::size_t n = weights.size();
    E c = E{};
    E rmse{};
    for (size_t i = 0; i < n; ++i) {
        E w = weights[i];
        assert(w >= 0);
        const E expected = weights[i] / *totalWeight;
        const E observed = static_cast<E>(histogram[i]) / n;
        const E diff = expected - observed;
        /* const E diff = (w * n - histogram[i] * totalWeight.value()) / (*totalWeight * n); */
        const E diff2 = diff * diff;
        const E term = diff2 - c;
        const E temp = rmse + term;
        c = (temp - rmse) - term;
        rmse = temp;
    }
    rmse = sqrt(rmse / static_cast<E>(weights.size()));
    return rmse;
}

template <std::floating_point E,
          std::integral I,
          std::ranges::input_range Scale,
          wrs::generic_allocator Allocator = std::allocator<E>>
    requires std::same_as<std::ranges::range_value_t<Scale>, I>
std::vector<std::tuple<I, E>,
            typename std::allocator_traits<Allocator>::template rebind_alloc<std::tuple<I, E>>>
rmse_curve(std::span<const E> weights,
           std::span<const I> samples,
           const Scale xScale,
           std::optional<E> totalWeight = std::nullopt,
           const Allocator& alloc = Allocator{}) {

    assert(!weights.empty());
    assert(!samples.empty());
    assert(!std::ranges::empty(xScale));

    using HistogramAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<I>;
    using ResultAllocator =
        typename std::allocator_traits<Allocator>::template rebind_alloc<std::tuple<I, E>>;

    if (!totalWeight.has_value()) {
        totalWeight = wrs::reference::kahan_reduction<E>(weights);
    }

    if (*totalWeight < std::numeric_limits<E>::epsilon()) {
        SPDLOG_WARN(fmt::format(
            "Computing rmse_curve with a very small total weight might lead to invalid results.\n",
            "totalWeight = {}", *totalWeight));
    }

    size_t scaleSize = 0;
    if constexpr (std::ranges::sized_range<Scale>) {
        scaleSize = std::ranges::size(xScale);
    } else {
        for (const auto& _ : xScale) {
            ++scaleSize;
        }
    }

    std::vector<I, HistogramAllocator> histogram(weights.size(), 0, alloc);

    std::vector<std::tuple<I, E>, ResultAllocator> results(alloc);
    results.reserve(scaleSize);

    I lastN = 0;

    for (const auto& n : xScale) {
        assert(n > 0 && "Sample size (n) must be greater than 0.");
        assert(n <= samples.size());
        assert(n >= lastN && "xScale must be monotone");

        for (I i = lastN; i < n; ++i) {
            assert(samples[i] < weights.size());
            histogram[samples[i]]++;
        }

        assert(std::accumulate(histogram.begin(), histogram.end(), 0) == n);

        lastN = n;

        E rmse = E{};
        E c = E{};

        E expectedAverage = static_cast<E>(n) / *totalWeight;

        for (size_t i = 0; i < weights.size(); ++i) {
            E w = weights[i];
            assert(w >= 0);
            const E expected = weights[i] / *totalWeight;
            const E observed = static_cast<E>(histogram[i]) / n;
            const E diff = expected - observed;
            /* const E diff = (w * n - histogram[i] * totalWeight.value()) / (*totalWeight * n); */
            const E diff2 = diff * diff;
            const E term = diff2 - c;
            const E temp = rmse + term;
            c = (temp - rmse) - term;
            rmse = temp;
        }
        rmse = sqrt(rmse / static_cast<E>(weights.size()));

        results.emplace_back(n, rmse);
    }

    return results;
}

namespace pmr {

template <std::floating_point E, std::integral I, std::ranges::input_range Scale>
    requires std::same_as<std::ranges::range_value_t<Scale>, I>
std::pmr::vector<std::tuple<I, E>>
rmse_curve(std::span<const E> weights,
           std::span<const I> samples,
           const Scale xScale,
           std::optional<E> totalWeight = std::nullopt,
           const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::eval::rmse_curve<E, I, Scale, std::pmr::polymorphic_allocator<void>>(
        weights, samples, xScale, totalWeight, alloc);
}
} // namespace pmr

} // namespace wrs::eval
