#pragma once

#include "src/host/why.hpp"
#include <concepts>
#include <span>
#include <vector>
namespace host {

template <std::integral T, host::typed_allocator<T> Allocator = std::allocator<T>>
std::vector<T> histogram(std::span<const T> samples, std::size_t N, const Allocator& alloc = {}) {
    std::vector<T> histogram(N, alloc);
    for (const auto& sample : samples) {
        histogram[sample] += 1;
    }
    return histogram;
}

} // namespace host
