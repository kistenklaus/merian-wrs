#pragma once

#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/test/histogram.hpp"
#include <cmath>
#include <optional>
#include <span>
namespace wrs::test {

template <std::floating_point E, std::integral I, wrs::typed_allocator<I> Allocator>
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

    std::vector<I, Allocator> observed = wrs::test::histogram(samples, weights.size(), alloc);

    E rmse2{};
    for (size_t i = 0; i < weights.size(); ++i) {
        const E expected = (weights[i] / totalWeight) * static_cast<E>(samples.size());
        const E observed = static_cast<E>(observed[i]);
        rmse2 += std::pow(observed - expected, 2);
    }
    const E rmse = std::sqrt(rmse2 / static_cast<E>(weights.size()));
    return rmse2;
}

} // namespace wrs::test
