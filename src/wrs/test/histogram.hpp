#pragma once

#include "src/wrs/why.hpp"
#include <algorithm>
#include <cassert>
#include <optional>
#include <span>
#include <vector>
namespace wrs::test {

template <std::integral I, wrs::typed_allocator<I> Allocator>
std::vector<I, Allocator> histogram(std::span<const I> samples,
                                    std::optional<std::size_t> populationSize = std::nullopt,
                                    const Allocator& alloc = {}) {
    assert(!samples.empty());
    if (!populationSize.has_value()) {
        auto it = std::max_element(samples.begin(), samples.end());
        assert(it != samples.end());
        populationSize = *it + 1;
    }
    assert(populationSize.has_value());

    std::vector<I, Allocator> histogram{*populationSize, alloc};
    for (const I& s : samples) {
        histogram[s]++;
    }
    return histogram;
}

} // namespace wrs::test
