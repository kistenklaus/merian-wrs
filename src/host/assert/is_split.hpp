#pragma once

#include "src/host/types/split.hpp"
#include "src/host/why.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cwctype>
#include <fmt/base.h>
#include <memory>
#include <ranges>
#include <sstream>
#include <utility>
#include <vector>
namespace host::test {

enum IsSplitErrorType : std::uint8_t {
    IS_SPLIT_ERROR_TYPE_NONE = 0x0,
    IS_SPLIT_ERROR_TYPE_INVALID_NUMBER_OF_SPLITS = 0x1,
    IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND = 0x2,
    IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND = 0x4,
    IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT = 0x8,
    IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT = 0x10,
    IS_SPLIT_ERROR_TYPE_INVALID_SPILL = 0x20,
};

template <arithmetic T, std::integral I> struct IsSplitIndexError {
    using Type = IsSplitErrorType;
    Type type;
    I index;
    Split<T, I> split;
    size_t n;
    T target;
    T sigma;
    T sigma2;

    IsSplitIndexError() = default;

    void appendMessageToStringStream(std::stringstream& ss, I N, I K) const {
        ss << "\t\tFailure at index = " << index << ": (err:" << static_cast<uint32_t>(type)
           << ")\n";
        ss << "\t\t\tGot = (" << split.i << ", " << split.j << ", " << split.spill << ")\n";
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
            ss << "\t\t\t\ti = " << split.i << ", j = " << split.j << ", n = " << n << "\n";
        }
        if (type & IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT) {
            ss << "\t\t\tBroken Invariant: " << sigma << " <= " << target << " && " << sigma2
               << " > " << target << " \n";
        }
        if (type & IS_SPLIT_ERROR_TYPE_INVALID_SPILL) {
            ss << "\t\t\tInvalid spill. Expected " << sigma2 - target << ", Got " << split.spill
               << "\n";
            ss << "\t\t\tsigma2 = " << sigma << ", target = " << target
               << ", diff = " << sigma2 - target << "\n";
        }
        if (type & IS_SPLIT_ERROR_TYPE_INVALID_NUMBER_OF_SPLITS) {
            ss << "\t\t\tInvalid error: Invalid number of splits\n";
        }
        if (type == IS_SPLIT_ERROR_TYPE_NONE) {
            ss << "\t\t\tInvalid error\n";
        }
    }
};

template <arithmetic T, std::integral I, generic_allocator Allocator> struct IsSplitError {
    using Type = IsSplitErrorType;
    using IndexError = IsSplitIndexError<T, I>;
    using allocator = std::allocator_traits<Allocator>::template rebind_alloc<IndexError>;
    static_assert(std::same_as<typename allocator::value_type, IndexError>);
    Type type;
    std::vector<IndexError, allocator> errors;
    I N;
    I K;
    I K_got;

    IsSplitError(Type type, std::vector<IndexError, allocator>&& errors, I N, I K, I K_got)
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

        if (type & IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT) {
            ss << "AssertionFailed: i + j != 0 (Somewhere)" << "\n";
        }
        if (type & IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND) {
            ss << "AssertionFailed: i is out of bounds (Somewhere)" << "\n";
        }

        if (type & IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND) {
            ss << "AssertionFailed: j is out of bounds (Somewhere)" << "\n";
        }

        ss << "AssertionFailed: At " << errors.size() << " out of " << K << "indicies\n";
        constexpr size_t MAX_LOG = 10;
        for (const auto& error : errors | std::views::take(MAX_LOG)) {
            error.appendMessageToStringStream(ss, N, K);
        }
        if (errors.size() > MAX_LOG) {
            ss << "\t...\n";
        }
        return ss.str();
    }
};

template <arithmetic T, std::integral I, generic_allocator Allocator = std::allocator<void>>
IsSplitError<
    T,
    I,
    typename std::allocator_traits<Allocator>::template rebind_alloc<IsSplitIndexError<T, I>>>
assert_is_split(std::span<const Split<T, I>> splits,
                I K,
                std::span<const T> heavyPrefix,
                std::span<const T> lightPrefix,
                const T mean,
                const T error_margin = {},
                const Allocator& alloc = {}) {
    using ErrorAllocator =
        std::allocator_traits<Allocator>::template rebind_alloc<IsSplitIndexError<T, I>>;
    /* static_assert(std::same_as<typename ErrorAllocator::value_type, IsSplitIndexError<T>>); */
    using Error = IsSplitError<T, I, ErrorAllocator>;
    using ErrorType = IsSplitErrorType;
    using IndexError = IsSplitIndexError<T, I>;
    I N = heavyPrefix.size() + lightPrefix.size();
    I K_got = splits.size();
    if (K_got != K) {
        std::vector<IndexError, ErrorAllocator> errors{ErrorAllocator(alloc)};
        return Error(IS_SPLIT_ERROR_TYPE_INVALID_NUMBER_OF_SPLITS, std::move(errors), N, K, K_got);
    }

    size_t errorCount = 0;
    for (I k = 1; k <= K - 1; ++k) {

        const auto& split = splits[k - 1];
        const auto& i = split.i;
        const auto& j = split.j;
        const auto& spill = split.spill;

        std::uint8_t type = IS_SPLIT_ERROR_TYPE_NONE;

        if (i - 1 >= lightPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND;
        }
        if (j >= heavyPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND;
        }
        const std::uintmax_t temp = static_cast<std::uintmax_t>(N) * static_cast<std::uintmax_t>(k);
        const I n = (temp + static_cast<std::uintmax_t>(K) - 1) / static_cast<std::uintmax_t>(K);

        if (i + j != n) {
            type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT;
        }
        // Invariant:
        // 1. sigma <= target
        // 2. sigma + next_heavy > target

        if (!((type & IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND) ||
              (type & IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND))) {

            const T target = mean * n;
            const T light = i == 0 ? 0 : lightPrefix[i - 1];
            const T heavy = j == 0 ? 0 : heavyPrefix[j - 1];
            const T heavy2 = heavyPrefix[j];
            const T sigma = light + heavy;
            const T sigma2 = light + heavy2;

            const T expectedSpill = sigma2 - target;
            if (std::abs(expectedSpill - spill) > 1e-5 ||
                std::abs((sigma2 - spill) - target) > 1e-5) {
                type |= IS_SPLIT_ERROR_TYPE_INVALID_SPILL;
            }

            if (!(sigma <= target && sigma2 > target)) {
                type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT;
            }
        }
        if (type != 0) {
            errorCount += 1;
        }
    }
    if ((splits.back().i != static_cast<I>(lightPrefix.size())) ||
        ((splits.back().j + 1) != static_cast<I>(heavyPrefix.size())) ||
        splits.back().spill > 0.01) {
        errorCount += 1;
    }

    std::vector<IsSplitIndexError<T, I>, ErrorAllocator> errors{errorCount, ErrorAllocator{alloc}};
    std::uint8_t aggType = IS_SPLIT_ERROR_TYPE_NONE;

    size_t e = 0;
    for (size_t k = 1; k <= K - 1; ++k) {

        const auto& split = splits[k - 1];
        const auto& i = split.i;
        const auto& j = split.j;
        const auto& spill = split.spill;

        std::uint8_t type = IS_SPLIT_ERROR_TYPE_NONE;

        if (i >= lightPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND;
        }
        if (j >= heavyPrefix.size()) {
            type |= IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND;
        }
        const std::uintmax_t temp = static_cast<std::uintmax_t>(N) * static_cast<std::uintmax_t>(k);
        const I n = (temp + static_cast<std::uintmax_t>(K) - 1) / static_cast<std::uintmax_t>(K);

        if (i + j != n) {
            type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIZE_INVARIANT;
        }

        // Invariant:
        // 1. sigma <= target
        // 2. sigma + next_heavy > target

        T target = 0;
        T sigma = 0;
        T sigma2 = 0;
        if (!((type & IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND) ||
              (type & IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND))) {

            target = mean * n;
            const T light = i == 0 ? 0 : lightPrefix[i - 1];
            const T heavy = j == 0 ? 0 : heavyPrefix[j - 1];
            const T heavy2 = heavyPrefix[j];
            sigma = light + heavy;
            sigma2 = light + heavy2;

            const T expectedSpill = sigma2 - target;
            if (std::abs(expectedSpill - spill) > 0.01 ||
                std::abs((sigma2 - spill) - target) > 0.01) {
                type |= IS_SPLIT_ERROR_TYPE_INVALID_SPILL;
            }

            if (!(sigma - target <= error_margin && error_margin > target - sigma2)) {
                type |= IS_SPLIT_ERROR_TYPE_BROKEN_SIGMA_INVARIANT;
            }
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
    if (splits.back().i != static_cast<I>(lightPrefix.size())) {
        type |= IS_SPLIT_ERROR_TYPE_I_OUT_OF_BOUND;
    }
    if (splits.back().j + 1 != static_cast<I>(heavyPrefix.size())) {
        type |= IS_SPLIT_ERROR_TYPE_J_OUT_OF_BOUND;
    }
    if (splits.back().spill > 0.01) {
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
template <arithmetic T, std::integral I>
IsSplitError<T, I, std::pmr::polymorphic_allocator<IsSplitIndexError<T, I>>>
assert_is_split(std::span<const Split<T, I>> splits,
                I K,
                std::span<const T> heavyPrefix,
                std::span<const T> lightPrefix,
                T mean,
                const T error_margin = {},
                const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return test::assert_is_split<T, I, std::pmr::polymorphic_allocator<void>>(
        splits, K, heavyPrefix, lightPrefix, mean, error_margin, alloc);
}
} // namespace pmr

} // namespace host::test
