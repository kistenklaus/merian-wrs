#pragma once

#include <concepts>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace wrs::test {

enum IsPrefixErrorType : unsigned int {
    IS_PREFIX_ERROR_TYPE_NONE = 0,
    IS_PREFIX_ERROR_TYPE_UNEQUAL_SIZE = 1,
    IS_PREFIX_ERROR_TYPE_NOT_MONOTONE = 2,
    IS_PREFIX_ERROR_TYPE_UNSTABLE = 4,
    IS_PREFIX_ERROR_TYPE_NOT_A_PREFIX_SUM = 8,
};

template <typename T> struct IsPrefixIndexError {
    using Type = IsPrefixErrorType;

    Type type = IS_PREFIX_ERROR_TYPE_NONE;
    size_t index = 0;
    T element = {};
    T prefix = {};
    T prevPrefix = {};
};

template <typename T, typename Allocator> struct IsPrefixError {
    using allocator =
        std::allocator_traits<Allocator>::template rebind_alloc<IsPrefixIndexError<T>>;
    using IError = IsPrefixIndexError<T>;
    using IErrorType = IsPrefixErrorType;
    const std::vector<IError, allocator> errors;
    const size_t elementSize;
    const size_t prefixSize;
    const IErrorType errorTypes;
    IsPrefixError() {}
    IsPrefixError(std::vector<IError, allocator> errors,
                  size_t elementSize,
                  size_t prefixSize,
                  IErrorType errorTypes)
        : errors(std::move(errors)), elementSize(elementSize), prefixSize(prefixSize),
          errorTypes(errorTypes) {}
    inline operator bool() {
        return !errors.empty() || elementSize != prefixSize;
    }
};

template <typename T, typename Allocator = std::allocator<void>>
IsPrefixError<
    T,
    typename std::allocator_traits<Allocator>::template rebind_alloc<IsPrefixIndexError<T>>>
assert_is_inclusive_prefix(const std::span<T> elements,
                           const std::span<T> prefix,
                           const Allocator& alloc = std::allocator<void>{}) {
    using Error = IsPrefixError<
        T, typename std::allocator_traits<Allocator>::template rebind_alloc<IsPrefixIndexError<T>>>;
    using allocator = Error::allocator;
    using IError = Error::IError;
    using IErrorType = Error::IErrorType;

    size_t esize = static_cast<size_t>(std::ranges::size(elements));
    size_t psize = static_cast<size_t>(std::ranges::size(prefix));
    if (esize != psize) {
        return Error(std::vector<IError, allocator>{alloc}, esize, psize,
                     IErrorType::IS_PREFIX_ERROR_TYPE_UNEQUAL_SIZE);
    }

    const auto eend = std::ranges::end(elements);
    const auto pend = std::ranges::end(prefix);

    T prevPrefix{}; // identity element
    size_t errorCount = 0;
    for (auto it1 = std::ranges::begin(elements), it2 = std::ranges::begin(prefix);
         it1 != eend && it2 != pend; ++it1, ++it2) {
        const T prefix = *it2;
        const T element = *it1;

        const auto diff = prevPrefix - prefix;
        if (element > 0) {
            if (diff <= 0) {
                errorCount += 1; // Not monotone!
                continue;
            }
        } else if (element < 0) {
            if (diff >= 0) {
                errorCount += 1; // Not monotone!
                continue;
            }
        }

        const T expectedDiff = std::abs(diff + element);
        if (expectedDiff > 0.01) {
            errorCount += 1;
            continue;
        }
        prevPrefix = prefix;
    }

    if (errorCount == 0) {
        return Error(std::vector<IError, allocator>{alloc}, esize, psize,
                     IErrorType::IS_PREFIX_ERROR_TYPE_NONE);
    }

    std::vector<IError, allocator> errors(errorCount, alloc);

    size_t i = 0;
    size_t j = 0;
    IErrorType allErrors = IErrorType::IS_PREFIX_ERROR_TYPE_NONE;
    for (auto it1 = std::ranges::begin(elements), it2 = std::ranges::begin(prefix);
         it1 != eend && it2 != pend; ++it1, ++it2, ++i) {
        const T prefix = *it2;
        const T element = *it1;

        IError error{
            .type = IS_PREFIX_ERROR_TYPE_NONE,
            .index = i,
            .element = element,
            .prefix = prefix,
            .prevPrefix = prevPrefix,
        };

        const auto diff = prevPrefix - prefix;
        if (element > 0) {
            if (diff <= 0) {
                error.type = static_cast<IErrorType>(
                    static_cast<unsigned int>(IErrorType::IS_PREFIX_ERROR_TYPE_NOT_MONOTONE) |
                    static_cast<unsigned int>(error.type));
            }
        } else if (element < 0) {
            if (diff >= 0) {
                error.type = static_cast<IErrorType>(
                    static_cast<unsigned int>(IErrorType::IS_PREFIX_ERROR_TYPE_NOT_MONOTONE) |
                    static_cast<unsigned int>(error.type));
            }
        }

        const T expectedDiff = std::abs(diff + element);
        if (expectedDiff > 1) {
            error.type = static_cast<IErrorType>(
                static_cast<unsigned int>(IErrorType::IS_PREFIX_ERROR_TYPE_NOT_A_PREFIX_SUM) |
                static_cast<unsigned int>(error.type));
        } else if (expectedDiff > 0.01) {
            error.type = static_cast<IErrorType>(
                static_cast<unsigned int>(IErrorType::IS_PREFIX_ERROR_TYPE_UNSTABLE) |
                static_cast<unsigned int>(error.type));
        }
        prevPrefix = prefix;

        if (error.type != IErrorType::IS_PREFIX_ERROR_TYPE_NONE) {
            allErrors = static_cast<IErrorType>(static_cast<unsigned int>(error.type) |
                                                static_cast<unsigned int>(allErrors));
            errors[j++] = error;
        }
    }
    return Error(errors, esize, psize, allErrors);
}

} // namespace wrs::test
