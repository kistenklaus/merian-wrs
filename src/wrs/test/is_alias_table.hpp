#pragma once

#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/why.hpp"
#include <concepts>
#include <memory>
#include <ranges>
#include <span>
#include <sstream>
#include <vector>
namespace wrs::test {

enum IsAliasTableErrorType : std::uint8_t {
    IS_ALIAS_TABLE_ERROR_TYPE_NONE = 0x0,
    IS_ALIAS_TABLE_ERROR_TYPE_INVALID_SIZE = 0x1,
    IS_ALIAS_TABLE_ERROR_TYPE_INVALID_ALIAS = 0x2,
    IS_ALIAS_TABLE_ERROR_TYPE_OVERSAMPLED_WEIGHT = 0x4,
    IS_ALIAS_TABLE_ERROR_TYPE_UNDERSAMPLED_WEIGHT = 0x8,
};

template <wrs::arithmetic T, std::floating_point P, std::integral I> struct IsAliasTableIndexError {
    using ErrorType = IsAliasTableErrorType;

    ErrorType type;
    T weight;
    P sampledWeight;
    I index;
    I alias;

    IsAliasTableIndexError() = default;

    void appendMessageToStringStream(std::stringstream& ss, I N) const {
        ss << "\tFailure at index: " << index << "\n";
        if (type & IS_ALIAS_TABLE_ERROR_TYPE_INVALID_ALIAS) {
            ss << "\t\tAlias (" << alias << ") out of bound. (Bound = " << N << ")\n";
        }
        if (type & IS_ALIAS_TABLE_ERROR_TYPE_OVERSAMPLED_WEIGHT) {
            ss << "\t\tWeight oversampled!\n";
            ss << "\t\t\tExpected: " << weight << ", Got: " << sampledWeight << "\n";
        }
        if (type & IS_ALIAS_TABLE_ERROR_TYPE_UNDERSAMPLED_WEIGHT) {
            ss << "\t\tWeight undersampled!\n";
            ss << "\t\t\tExpected: " << weight << ", Got: " << sampledWeight << "\n";
        }
    }
};

template <wrs::arithmetic T, std::floating_point P, std::integral I, wrs::generic_allocator Allocator> struct IsAliasTableError {
    using IndexError = IsAliasTableIndexError<T, P, I>;
    using ErrorType = IsAliasTableErrorType;

    using allocator = std::allocator_traits<Allocator>::template rebind_alloc<IndexError>;

    ErrorType type;
    std::vector<IndexError, allocator> errors;
    I N;
    I N_got;
    P maxOversample;
    P maxUndersample;

    IsAliasTableError(ErrorType type,
                      std::vector<IndexError, allocator>&& errors,
                      I N,
                      I N_got, P maxOversample = {},
                      P maxUndersample = {})
        : type(type), errors(std::move(errors)), N(N), N_got(N_got), maxOversample(maxOversample),
    maxUndersample(maxUndersample){}

    operator bool() const {
        return type != IS_ALIAS_TABLE_ERROR_TYPE_NONE;
    }

    std::string message() const {
        std::stringstream ss;
        if (type & IS_ALIAS_TABLE_ERROR_TYPE_INVALID_SIZE) {
            ss << "AssertionFailed: The size of the alias table is incorrect:\n";
            ss << "\tExpected: " << N << ", Got: " << N_got << "\n";
            return ss.str();
        }
        ss << "AssertionFailed: At " << errors.size() << " out of " << N << " indicies\n";
        ss << "\tWith failures:\n";
        if (type & IS_ALIAS_TABLE_ERROR_TYPE_INVALID_ALIAS) {
            ss << "\t\t-INVALID_ALIAS\n";
        }
        if (type & IS_ALIAS_TABLE_ERROR_TYPE_UNDERSAMPLED_WEIGHT) {
            ss << "\t\t-UNDERSAMPLED_WEIGHT: worstCase = " << maxUndersample << "\n";
        }
        if (type & IS_ALIAS_TABLE_ERROR_TYPE_OVERSAMPLED_WEIGHT) {
            ss << "\t\t-OVERSAMPLED_WEIGHT : worstCase = "<< maxOversample << "\n";
        }

        constexpr std::size_t MAX_LOG = 10;
        for (const auto& error : errors | std::views::take(MAX_LOG)) {
            error.appendMessageToStringStream(ss, N);
        }
        if (errors.size() > MAX_LOG) {
            ss << "\t...\n";
        }
        return ss.str();
    };
};

template <wrs::arithmetic T, std::floating_point P, std::integral I, wrs::generic_allocator Allocator>
IsAliasTableError<
    T,
    P,
    I,
    typename std::allocator_traits<Allocator>::template rebind_alloc<IsAliasTableIndexError<T, P, I>>>
assert_is_alias_table(std::span<const T> weights,
                      std::span<const wrs::AliasTableEntry<P, I>> aliasTable,
                      const T totalWeight,
                      const P errorMargin = 0.001,
                      const Allocator& alloc = {}) {
    using IndexError = IsAliasTableIndexError<T, P, I>;
    using ErrorType = IsAliasTableErrorType;
    using ErrorAllocator = std::allocator_traits<Allocator>::template rebind_alloc<IndexError>;
    using Error = IsAliasTableError<T, P, I, ErrorAllocator>;
    using PAllocator = std::allocator_traits<Allocator>::template rebind_alloc<P>;

    if (weights.size() != aliasTable.size()) {
        std::vector<IndexError, ErrorAllocator> errors;
        return Error(IS_ALIAS_TABLE_ERROR_TYPE_INVALID_SIZE, std::move(errors), weights.size(),
                     aliasTable.size());
    }

    // Compute weights sampled by the alias table!
    std::vector<P, PAllocator> sampled{aliasTable.size(), PAllocator{alloc}};
    for (std::size_t i = 0; i < aliasTable.size(); ++i) {
        const auto& entry = aliasTable[i];
        sampled[i] += entry.p;
        if (entry.a < sampled.size()) {
            sampled[entry.a] += (1.0f - entry.p);
        }
    }
    /* P norm = static_cast<P>(reduction) / static_cast<P>(sampled.size()); */
    // Normalize sampled to probabilties * totalWeight (Should be decently stable )
    const P totalWeightP = static_cast<P>(totalWeight);
    const P sampledSize = static_cast<P>(sampled.size());
    for (std::size_t i = 0; i < sampled.size(); ++i) {
        // Prevent floating point reordering under -ffast-math (i.e -Ofast)
        volatile P s = sampled[i];
        s *= totalWeightP;
        s /= sampledSize;
        sampled[i] = s;
    }

    std::size_t errorCount = 0;
    P maxOversample = 0;
    P maxUndersample = 0;
    for (size_t i = 0; i < aliasTable.size(); ++i) {
        std::uint8_t type = IS_ALIAS_TABLE_ERROR_TYPE_NONE;
        const auto& entry = aliasTable[i];
        if (entry.a >= aliasTable.size()) {
            type |= IS_ALIAS_TABLE_ERROR_TYPE_INVALID_ALIAS;
        }
        if (sampled[i] - errorMargin > weights[i]) {
            maxOversample = std::max(sampled[i] - weights[i], maxOversample);
            type |= IS_ALIAS_TABLE_ERROR_TYPE_OVERSAMPLED_WEIGHT;
        } else if (sampled[i] + errorMargin < weights[i]) {
            maxUndersample = std::max(weights[i] - sampled[i], maxUndersample);
            type |= IS_ALIAS_TABLE_ERROR_TYPE_UNDERSAMPLED_WEIGHT;
        }
        if (type != IS_ALIAS_TABLE_ERROR_TYPE_NONE) {
            errorCount += 1;
        }
    }

    std::uint8_t aggError = 0;
    std::vector<IndexError, ErrorAllocator> errors{errorCount, ErrorAllocator{alloc}};
    std::size_t e = 0;
    for (size_t i = 0; i < aliasTable.size(); ++i) {
        std::uint8_t type = IS_ALIAS_TABLE_ERROR_TYPE_NONE;
        const auto& entry = aliasTable[i];
        if (entry.a >= aliasTable.size()) {
            type |= IS_ALIAS_TABLE_ERROR_TYPE_INVALID_ALIAS;
        }
        if (sampled[i] - errorMargin > weights[i]) {
            type |= IS_ALIAS_TABLE_ERROR_TYPE_OVERSAMPLED_WEIGHT;
        } else if (sampled[i] + errorMargin < weights[i]) {
            type |= IS_ALIAS_TABLE_ERROR_TYPE_UNDERSAMPLED_WEIGHT;
        }
        if (type != IS_ALIAS_TABLE_ERROR_TYPE_NONE) {
            aggError |= type;
            errors[e].type = static_cast<ErrorType>(type);
            errors[e].weight = weights[i];
            errors[e].sampledWeight = sampled[i];
            errors[e].index = i;
            errors[e].alias = entry.a;
            e++;
        }
    }

    return Error(static_cast<ErrorType>(aggError), std::move(errors), weights.size(),
                 aliasTable.size(), maxOversample, maxUndersample);
}

namespace pmr {
template <wrs::arithmetic T, std::floating_point P, std::integral I>
IsAliasTableError<T, P, I, std::pmr::polymorphic_allocator<IsAliasTableIndexError<T, P, I>>>
assert_is_alias_table(std::span<const T> weights,
                      std::span<const wrs::AliasTableEntry<P, I>> aliasTable,
                      const T totalWeight,
                      const P errorMargin = 0.001,
                      const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::test::assert_is_alias_table<T, P, I, std::pmr::polymorphic_allocator<void>>(
        weights, aliasTable, totalWeight, errorMargin, alloc);
}
} // namespace pmr

} // namespace wrs::test
