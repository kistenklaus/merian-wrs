
#include <concepts>
#include <cstdint>
#include <memory_resource>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>
namespace wrs::reference {

template <typename T, typename SizedRange, typename Allocator = std::allocator<T>>
    requires std::same_as<T, std::ranges::range_value_t<SizedRange>> &&
             std::ranges::sized_range<SizedRange>
std::vector<T, Allocator> prefix_sum(const SizedRange weights, const Allocator& alloc) {
    std::vector<T, Allocator> prefix(weights.begin(), weights.end(), alloc);
    uint64_t N = std::ranges::size(weights);
    for (uint64_t shift = 1; shift <= N; shift <<= 1) {
        for (size_t i = N - 1; i >= shift; --i) {
            prefix.at(i) += prefix.at(i - shift);
        }
    }
    return prefix;
}

namespace pmr {

template <typename T, typename SizedRange>
    requires std::same_as<T, std::ranges::range_value_t<SizedRange>> &&
             std::ranges::sized_range<SizedRange>
std::pmr::vector<T> prefix_sum(const SizedRange& weights,
                               const std::pmr::polymorphic_allocator<T>& alloc) {
    return wrs::reference::prefix_sum<T, SizedRange, std::pmr::polymorphic_allocator<T>>(weights,
                                                                                         alloc);
}

} // namespace pmr

} // namespace wrs::reference
