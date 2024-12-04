#pragma once

#include <cstdint>
#include <span>
#include <tuple>
#include <vector>
namespace wrs::test {

namespace internal {
template <typename T> using Split = std::tuple<std::size_t, std::size_t, T>;

}

enum IsSplitErrorType : std::uint8_t {
    IS_SPLIT_ERROR_TYPE_NONE = 0,
    IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND = 1,
    IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND = 2,
    IS_SPLIT_ERROR_TYPE_BROKEN_INVARIANT = 4,
};

template <typename T> struct IsSplitIndexError {
    using Type = IsSplitErrorType;
    Type type;
    std::size_t index;
    internal::Split<T> split;
};

template <typename T, typename Allocator = std::allocator<IsSplitIndexError<T>>>
struct IsSplitError {
    using Type = IsSplitErrorType;
    Type type;
    std::vector<IsSplitIndexError<T>, Allocator> errors;

    operator bool() const {
        return type == IS_SPLIT_ERROR_TYPE_NONE;
    }
};

template <typename T, typename Allocator = std::allocator<void>>
IsSplitError<T,
             typename std::allocator_traits<Allocator>::template rebind_alloc<IsSplitIndexError<T>>>
assert_is_split(std::span<internal::Split<T>> splits,
                std::span<T> heavyPrefix,
                std::span<T> lightPrefix,
                T mean,
                const Allocator& alloc = {}) {
    using ErrorAllocator =
        std::allocator_traits<Allocator>::template rebind_alloc<IsSplitIndexError<T>>;

    size_t K = splits.size();
    size_t N = heavyPrefix.size() + lightPrefix.size();

    size_t errorCount = 0;
    for (size_t k = 1; k < K; ++k) {
        const auto& [i, j, spill] = splits[k - 1];

        std::uint8_t type = IS_SPLIT_ERROR_TYPE_NONE;

        if (i < lightPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND;
        }
        if (j < heavyPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND;
        }
        // Invariant:
        // 1. sigma <= target
        // 2. sigma + next_heavy > target

        const T temp = N * (k + 1);
        const T target = 1 + ((temp - 1) / K);
        const T sigma = lightPrefix[i] + heavyPrefix[j];
        const T sigma2 = lightPrefix[i] + heavyPrefix[j + 1];
        if (sigma <= target && sigma2 > target) {

        } else {
            type |= IS_SPLIT_ERROR_TYPE_BROKEN_INVARIANT;
        }

        if (type != 0) {
            errorCount += 1;
        }
    }

    std::vector<IsSplitIndexError<T>, ErrorAllocator> errors{errorCount, ErrorAllocator(alloc)};

    return IsSplitError<T, Allocator>{
        .type = {},
        .errors = std::move(errors),
    };
}

namespace pmr {
template <typename T>
IsSplitError<T, std::pmr::polymorphic_allocator<void>>
assert_is_split(std::span<internal::Split<T>> splits,
                std::span<T> heavyPrefix,
                std::span<T> lightPrefix,
                T mean,
                const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return std::move(wrs::test::assert_is_split<T, std::pmr::polymorphic_allocator<void>>(
        splits, heavyPrefix, lightPrefix, mean, alloc));
}
} // namespace pmr

} // namespace wrs::test
