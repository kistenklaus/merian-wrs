
#include <concepts>
#include <cstdint>
#include <memory_resource>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>
namespace wrs::reference {

template <typename T, typename Allocator = std::allocator<T>>
std::vector<T, Allocator>
prefix_sum(const std::span<T> elements, bool ensureMonotone = true, const Allocator& alloc = {}) {
    std::vector<T, Allocator> prefix(elements.begin(), elements.end(), alloc);
    uint64_t N = std::ranges::size(elements);
    // Inital not work efficient algorithm:
    // - Reasonable numerical stability errors should not accumulate that much
    // - Problem: Reuslt is not guaranteed to be monotone when working with floating point numbers
    for (uint64_t shift = 1; shift <= N; shift <<= 1) {
        for (size_t i = N - 1; i >= shift; --i) {
            prefix.at(i) += prefix.at(i - shift);
        }
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

template <typename T>
std::pmr::vector<T> prefix_sum(const std::span<T> weights,
                               bool ensureMonotone = true,
                               const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    return wrs::reference::prefix_sum<T, std::pmr::polymorphic_allocator<T>>(weights,
                                                                             ensureMonotone, alloc);
}

} // namespace pmr

} // namespace wrs::reference
