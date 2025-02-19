#pragma once

#include "src/host/why.hpp"
#include <memory>
#include <ranges>
#include <type_traits>
#include <vector>

namespace host::test {

enum class ErrorType {
    UNEQUAL_SIZES,
    UNEQUAL_SIMILAR_VALUES,
    UNEQUAL_UNSIMILAR_VALUES,
};

template <typename A, typename B, typename SA, typename SB, typename Allocator>
struct CompareResults {
    using DiffType = std::conditional<sizeof(A) >= sizeof(B), A, B>;
    const ErrorType type;
    SA a_size;
    SB b_size;
    std::vector<std::tuple<size_t, A, B>, Allocator> unequal_values;
    CompareResults(ErrorType ty,
                   SA a_size,
                   SB b_size,
                   std::vector<std::tuple<size_t, A, B>, Allocator> values)
        : type(ty) {
        switch (ty) {
        case ErrorType::UNEQUAL_SIZES:
            this->a_size = a_size;
            this->b_size = b_size;
            break;
        case ErrorType::UNEQUAL_SIMILAR_VALUES:
        case ErrorType::UNEQUAL_UNSIMILAR_VALUES:
            this->unequal_values = std::move(values);
            break;
        }
    }
    ~CompareResults() = default;
};

template <std::ranges::sized_range R1,
          std::ranges::sized_range R2,
          generic_allocator ErrorAllocator = std::allocator<void>>
std::unique_ptr<CompareResults<std::ranges::range_value_t<R1>,
                               std::ranges::range_value_t<R2>,
                               std::ranges::range_size_t<R1>,
                               std::ranges::range_size_t<R2>,
                               ErrorAllocator>>
compare_ranges(const R1& r1, const R2& r2, const ErrorAllocator& alloc = {}) {
    using A = std::ranges::range_value_t<R1>;
    using B = std::ranges::range_value_t<R2>;
    using size1_t = std::ranges::range_size_t<R1>;
    using size2_t = std::ranges::range_size_t<R2>;
    using Result = CompareResults<A, B, size1_t, size2_t, ErrorAllocator>;

    size1_t size1 = std::ranges::size(r1);
    size2_t size2 = std::ranges::size(r2);

    if (size1 != size2) {
        return Result(ErrorType::UNEQUAL_SIZES, size1, size2, {});
    }

    const auto end1 = std::ranges::end(r1);
    const auto end2 = std::ranges::end(r2);
    bool equal = true;
    bool similar = true;
    size_t unequalValues;
    for (auto it1 = std::ranges::begin(r1), it2 = std::ranges::begin(r2);
         it1 != end1 && it2 != end2; ++it1, ++it2) {
        const auto a = *it1;
        const auto b = *it2;
        if (a == b) {
            equal = true;
            unequalValues += 1;
        }
        if constexpr (std::is_floating_point_v<A> || std::is_floating_point_v<B>) {
            const auto quotDiff = std::abs(std::abs(a / b) - 1.0f);
            if (quotDiff > 0.001) {
                similar = false;
            }

        } else {
            const auto diff = std::abs(a - b);
            if (diff > 0.01) {
                similar = false;
            }
        }
    }
    if (equal) {
        return nullptr;
    }
    using Allocator =
        std::allocator_traits<ErrorAllocator>::template rebind_alloc<std::tuple<size_t, A, B>>;
    std::vector<std::tuple<size_t, A, B>, ErrorAllocator> unequalValuesVec(unequalValues,
                                                                           Allocator(alloc));
    auto it1 = std::ranges::begin(r1);
    auto it2 = std::ranges::begin(r2);
    for (size_t i = 0; i < unequalValues; ++it1, ++it2, ++i) {
        const auto a = *it1;
        const auto b = *it2;
        if (a == b) {
            unequalValuesVec.emplace_back(i, a, b);
        }
    }

    return Result(similar ? ErrorType::UNEQUAL_SIMILAR_VALUES : ErrorType::UNEQUAL_UNSIMILAR_VALUES,
                  size1, size2, std::move(unequalValuesVec));
}

} // namespace wrs::test
