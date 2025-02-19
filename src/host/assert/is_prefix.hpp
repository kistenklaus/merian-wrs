#pragma once

#include "src/host/why.hpp"
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <ranges>
#include <spdlog/spdlog.h>
#include <sstream>
#include <vector>

namespace host::test {

enum IsPrefixErrorType : unsigned int {
    IS_PREFIX_ERROR_TYPE_NONE = 0,
    IS_PREFIX_ERROR_TYPE_UNEQUAL_SIZE = 1,
    IS_PREFIX_ERROR_TYPE_NOT_MONOTONE = 2,
    IS_PREFIX_ERROR_TYPE_UNSTABLE = 4,
    IS_PREFIX_ERROR_TYPE_NOT_A_PREFIX_SUM = 8,
};

template <arithmetic T> struct IsPrefixIndexError {
    using Type = IsPrefixErrorType;

    Type type = IS_PREFIX_ERROR_TYPE_NONE;
    size_t index = 0;
    T element = {};
    T prefix = {};
    T prevPrefix = {};

    void appendMessageToStringStream(std::stringstream& ss) {
        ss << "\t\tFailure at index = " << index << ":\n";
        if (type & IS_PREFIX_ERROR_TYPE_NOT_MONOTONE) {
            ss << "\t\t\tPrefix is not monotone:\n";
            ss << "\t\t\t\tprefix[" << index << " - 1] = " << fmt::format("{:.10}", prevPrefix)
               << "\n";
            ss << "\t\t\t\t    prefix[" << index << "] = " << fmt::format("{:.10}", prefix)
               << "\t\t Diff : " << prefix - prevPrefix << "\n";
            ss << "\t\t\t\t   element[" << index << "] = " << element << "\n";
        }
        if (type & IS_PREFIX_ERROR_TYPE_UNSTABLE) {
            ss << "\t\t\tPrefix is numerically unstable:\n";
            ss << std::fixed << std::setprecision(6);
            ss << "\t\t\t\tExpected: " << prevPrefix << " + " << element << " \u2248 "
               << prevPrefix + element << "\t\t" << "Diff: |"
               << std::abs((prevPrefix - prefix) + element) << "|\n";
            ss << "\t\t\t\t     Got: " << prefix << std::endl;
        }
        if (type & IS_PREFIX_ERROR_TYPE_NOT_A_PREFIX_SUM) {
            ss << "\t\t\tPrefix is not really a prefix sum:\n";
            ss << "\t\t\t\tprefix[" << index << " - 1] != " << prevPrefix << "  "
               << prefix + element << " = prefix[" << index << "] + element[index]\n";
        }
    }
};

template <arithmetic T, generic_allocator Allocator> struct IsPrefixError {
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
    operator bool() const {
        return !errors.empty() || elementSize != prefixSize;
    }

    std::string message() const {
        std::stringstream ss;
        if (errorTypes & IS_PREFIX_ERROR_TYPE_UNEQUAL_SIZE) {
            ss << "AssertionFailed: The size of the prefix sum is incorrect:\n";
            ss << "\tExpected: " << elementSize << ", Got: " << prefixSize << "\n";
            return ss.str();
        }

        ss << "AssertionFailed: \"prefix\" is not a prefix sum.\n";
        ss << "\tFailed at " << errors.size() << " out of " << elementSize << " elements\n";
        constexpr size_t MAX_LOG = 3;
        for (auto error : errors | std::views::take(MAX_LOG)) {
            error.appendMessageToStringStream(ss);
        }
        if (errors.size() > MAX_LOG) {
            ss << "\t...\n";
        }
        return ss.str();
    }
};

template <arithmetic T, generic_allocator Allocator = std::allocator<void>>
IsPrefixError<
    T,
    typename std::allocator_traits<Allocator>::template rebind_alloc<IsPrefixIndexError<T>>>
assert_is_inclusive_prefix(std::span<const T> elements,
                           std::span<const T> prefix,
                           const Allocator& alloc = std::allocator<void>{}) {
    using Error = IsPrefixError<
        T, typename std::allocator_traits<Allocator>::template rebind_alloc<IsPrefixIndexError<T>>>;
    using allocator = Error::allocator;
    using IError = Error::IError;
    using IErrorType = Error::IErrorType;

    size_t esize = elements.size();
    size_t psize = prefix.size();
    if (esize != psize) {
        return Error(std::vector<IError, allocator>{alloc}, esize, psize,
                     IErrorType::IS_PREFIX_ERROR_TYPE_UNEQUAL_SIZE);
    }

    const auto eend = std::ranges::end(elements);
    const auto pend = std::ranges::end(prefix);

    T prevPrefix{}; // identity element
    size_t errorCount = 0;
    bool first = true;
    const T unstableMargin = 0.1;

    const T bullshitMargin = unstableMargin * 10;
    for (auto it1 = std::ranges::begin(elements), it2 = std::ranges::begin(prefix);
         it1 != eend && it2 != pend; ++it1, ++it2) {
        const T prefix = *it2;
        const T element = *it1;

        const auto diff = prevPrefix - prefix;
        if (!first) {
            if (element > 0) {
                if (diff > 0 && prevPrefix > prefix) {
                    errorCount += 1; // Not monotone!
                    prevPrefix = prefix;
                    continue;
                }
            } else if (element < 0) {
                if (diff < 0 && prevPrefix < prefix) {
                    errorCount += 1; // Not monotone!
                    prevPrefix = prefix;
                    continue;
                }
            }
        }
        first = false;
        const T expectedDiff = std::abs(diff + element);
        if (expectedDiff > unstableMargin) {
            errorCount += 1;
            prevPrefix = prefix;
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
    prevPrefix = {};
    IErrorType allErrors = IErrorType::IS_PREFIX_ERROR_TYPE_NONE;
    first = true;
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
        error.type = IErrorType::IS_PREFIX_ERROR_TYPE_NONE;

        const auto diff = prevPrefix - prefix;
        if (!first) {
            if (element > 0) {
                if (diff > 0 && prevPrefix > prefix) {
                    error.type = static_cast<IErrorType>(
                        static_cast<unsigned int>(IErrorType::IS_PREFIX_ERROR_TYPE_NOT_MONOTONE) |
                        static_cast<unsigned int>(error.type));
                }
            } else if (element < 0) {
                if (diff < 0 && prevPrefix < prefix) {
                    error.type = static_cast<IErrorType>(
                        static_cast<unsigned int>(IErrorType::IS_PREFIX_ERROR_TYPE_NOT_MONOTONE) |
                        static_cast<unsigned int>(error.type));
                }
            }
        }

        first = false;

        const T expectedDiff = std::abs(diff + element);
        if (expectedDiff > bullshitMargin) {
            error.type = static_cast<IErrorType>(
                static_cast<unsigned int>(IErrorType::IS_PREFIX_ERROR_TYPE_NOT_A_PREFIX_SUM) |
                static_cast<unsigned int>(error.type));
        } else if (expectedDiff > unstableMargin) {
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
    return Error(std::move(errors), esize, psize, allErrors);
}

namespace pmr {

template <arithmetic T>
IsPrefixError<T, std::pmr::polymorphic_allocator<IsPrefixIndexError<T>>>
assert_is_inclusive_prefix(std::span<const T> elements,
                           std::span<const T> prefix,
                           const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return test::assert_is_inclusive_prefix<T, std::pmr::polymorphic_allocator<void>>(
        elements, prefix, alloc);
}
} // namespace pmr

// LOGGING

} // namespace host::test
