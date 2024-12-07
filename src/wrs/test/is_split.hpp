#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cwctype>
#include <fmt/base.h>
#include <memory>
#include <ranges>
#include <span>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>
namespace wrs::test {

namespace internal {
template <typename T> using Split = std::tuple<std::size_t, std::size_t, T>;

}

enum IsSplitErrorType : std::uint8_t {
    IS_SPLIT_ERROR_TYPE_NONE = 0x0,
    IS_SPLIT_ERROR_TYPE_INVALID_NUMBER_OF_SPLITS = 0x1,
    IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND = 0x2,
    IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND = 0x4,
    IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT = 0x8,
    IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT = 0x10,
};

template <typename T> struct IsSplitIndexError {
    using Type = IsSplitErrorType;
    Type type;
    std::size_t index;
    internal::Split<T> split;
    size_t n;
    T target;
    T sigma;
    T sigma2;

    IsSplitIndexError() = default;

    void appendMessageToStringStream(std::stringstream& ss, std::size_t N, std::size_t K) const {
        ss << "\t\tFailure at index = " << index << ":\n";
        ss << "\t\t\tGot = (" << std::get<0>(split) << ", " << std::get<1>(split) << ", "
           << std::get<2>(split) << ")\n";
        if (type & IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND) {
            if (index != K - 1) {
                ss << "\t\t\ti out of bound. N = " << N << "\n";
            } else {
                ss << "\t\t\tExpected i = N = " << N << "\n";
            }
        }
        if (type & IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND) {
            if (index != K - 1) {
                ss << "\t\t\tj out of bound. N = " << N << "\n";
            } else {
                ss << "\t\t\tExpected i = N = " << N << "\n";
            }
        }
        if (type & IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT) {
            ss << "\t\t\tBroken Invariant: i + j = n\n";
            ss << "\t\t\t\ti = " << std::get<0>(split) << ", j = " << std::get<1>(split)
               << ", n = " << n << "\n";
        }
        if (type & IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT) {
            ss << "\t\t\tBroken Invariant: " << sigma << " <= " << target << " && " << sigma2
               << " > " << target << " \n";
        }
    }
};

template <typename T, typename Allocator = std::allocator<IsSplitIndexError<T>>>
struct IsSplitError {
    using Type = IsSplitErrorType;
    using IndexError = IsSplitIndexError<T>;
    using allocator = std::allocator_traits<Allocator>::template rebind_alloc<IndexError>;
    static_assert(std::same_as<typename allocator::value_type, IndexError>);
    Type type;
    std::vector<IndexError, allocator> errors;
    std::size_t N;
    std::size_t K;
    std::size_t K_got;

    IsSplitError(Type type,
                 std::vector<IndexError, allocator>&& errors,
                 std::size_t N,
                 std::size_t K,
                 std::size_t K_got)
        : type(type), errors(std::move(errors)), N(N), K(K), K_got(K_got) {}

    operator bool() const {
        return type != IS_SPLIT_ERROR_TYPE_NONE;
    }

    std::string message() const {
        std::stringstream ss{};
        if (type & IS_SPLIT_ERROR_TYPE_INVALID_NUMBER_OF_SPLITS) {
            ss << "AssertionFailed: Invalid number of splits\n";
            ss << "\tExpected = " << K << ", Got = " << K_got << "\n";
            return ss.str();
        }

        ss << "AssertionFailed: At " << errors.size() << " out of " << K << "indicies\n";
        constexpr size_t MAX_LOG = 3;
        for (const auto& error : errors | std::views::take(MAX_LOG)) {
            error.appendMessageToStringStream(ss, N, K);
        }
        if (errors.size() > MAX_LOG) {
            ss << "\t...\n";
        }
        return ss.str();
    }
};

template <typename T, typename Allocator = std::allocator<void>>
IsSplitError<T,
             typename std::allocator_traits<Allocator>::template rebind_alloc<IsSplitIndexError<T>>>
assert_is_split(std::span<internal::Split<T>> splits,
                std::size_t K,
                std::span<T> heavyPrefix,
                std::span<T> lightPrefix,
                const T mean,
                const T error_margin = {},
                const Allocator& alloc = {}) {
    using ErrorAllocator =
        std::allocator_traits<Allocator>::template rebind_alloc<IsSplitIndexError<T>>;
    /* static_assert(std::same_as<typename ErrorAllocator::value_type, IsSplitIndexError<T>>); */
    using Error = IsSplitError<T, ErrorAllocator>;
    using ErrorType = IsSplitErrorType;
    using IndexError = IsSplitIndexError<T>;
    size_t N = heavyPrefix.size() + lightPrefix.size();
    size_t K_got = splits.size();
    if (K_got != K) {
        std::vector<IndexError, ErrorAllocator> errors{ErrorAllocator(alloc)};
        return Error(IS_SPLIT_ERROR_TYPE_INVALID_NUMBER_OF_SPLITS, std::move(errors), N, K, K_got);
    }

    size_t errorCount = 0;
    for (size_t k = 1; k <= K - 1; ++k) {

        const auto& [i, j, spill] = splits[k - 1];

        std::uint8_t type = IS_SPLIT_ERROR_TYPE_NONE;

        if (i >= lightPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND;
        }
        if (j + 1 >= heavyPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND;
        }
        const size_t temp = N * k;
        const size_t n = 1 + ((temp - 1) / K);
        if (i + j != n) {
            type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT;
        }
        // Invariant:
        // 1. sigma <= target
        // 2. sigma + next_heavy > target

        const T target = mean * n;
        const T sigma = lightPrefix[i] + heavyPrefix[j];
        const T sigma2 = lightPrefix[i] + heavyPrefix[j + 1];
        if (!(sigma <= target && sigma2 > target)) {
            type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT;
        }
        if (type != 0) {
            errorCount += 1;
        }
    }
    if (std::get<0>(splits.back()) != N && std::get<1>(splits.back()) != N &&
        std::get<2>(splits.back()) > 0.01) {
        errorCount += 1;
    }

    std::vector<IsSplitIndexError<T>, ErrorAllocator> errors{errorCount, ErrorAllocator{alloc}};
    std::uint8_t aggType = IS_SPLIT_ERROR_TYPE_NONE;

    size_t e = 0;
    for (size_t k = 1; k <= K - 1; ++k) {

        const auto& [i, j, spill] = splits[k - 1];

        std::uint8_t type = IS_SPLIT_ERROR_TYPE_NONE;

        if (i >= lightPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND;
        }
        if (j + 1 >= heavyPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND;
        }
        const size_t temp = N * k;
        const size_t n = 1 + ((temp - 1) / K);
        if (i + j != n) {
            type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT;
        }

        // Invariant:
        // 1. sigma <= target
        // 2. sigma + next_heavy > target
        const T target = mean * n;
        const T sigma = lightPrefix[i] + heavyPrefix[j];
        const T sigma2 = lightPrefix[i] + heavyPrefix[j + 1];

        if (!(sigma - target <= error_margin && error_margin > target - sigma2)) {
            type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT;
        }

        if (type != 0) {
            aggType |= type;
            errors[e].type = static_cast<IsSplitErrorType>(type);
            errors[e].index = k - 1;
            errors[e].split = splits[k - 1];
            errors[e].n = n;
            errors[e].target = target;
            errors[e].sigma = sigma;
            errors[e].sigma2 = sigma2;
            e += 1;
        }
    }
    std::uint8_t type = IS_SPLIT_ERROR_TYPE_NONE;
    if (std::get<0>(splits.back()) != N) {
        type |= IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND;
    }
    if (std::get<1>(splits.back()) != N) {
        type |= IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND;
    }
    if (std::get<2>(splits.back()) > 0.01) {
        type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT;
    }
    if (type != IS_SPLIT_ERROR_TYPE_NONE) {
        aggType |= type;
        errors[e].type = static_cast<IsSplitErrorType>(type);
        errors[e].index = K - 1;
        errors[e].split = splits.back();
        errors[e].n = N;
        errors[e].target = 0;
        errors[e].sigma = 0;
        errors[e].sigma2 = 0;
        e += 1;
    }

    return Error(static_cast<ErrorType>(aggType), std::move(errors), N, K, K_got);
}

namespace pmr {
template <typename T>
IsSplitError<T, std::pmr::polymorphic_allocator<IsSplitIndexError<T>>>
assert_is_split(std::span<internal::Split<T>> splits,
                std::size_t K,
                std::span<T> heavyPrefix,
                std::span<T> lightPrefix,
                T mean,
                const T error_margin = {},
                const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::test::assert_is_split<T, std::pmr::polymorphic_allocator<void>>(
        splits, K, heavyPrefix, lightPrefix, mean, error_margin, alloc);
}
} // namespace pmr

} // namespace wrs::test
