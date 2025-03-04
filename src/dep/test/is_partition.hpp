#pragma once

#include "src/wrs/why.hpp"
#include <algorithm>
#include <concepts>
#include <memory>
#include <ranges>
#include <span>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>
namespace wrs::test {

enum IsPartitionErrorType : unsigned int {
    IS_PARTITION_ERROR_TYPE_NONE = 0,
    IS_PARTITION_ERROR_TYPE_INVALID_PARTITION_SIZES = 1,
    IS_PARTITION_ERROR_TYPE_INVALID_PARTITION = 2, // element assigned to the wrong partition
    IS_PARTITION_ERROR_TYPE_INVALID_ELEMENT = 4,
};

template <std::totally_ordered T> struct IsPartitionIndexError {
    IsPartitionErrorType type;
    size_t index;
    size_t elementIndex;
    bool shouldBeHeavy;
    bool isHeavy;
    T value;

    void appendMessageToStringStream(std::stringstream& ss) {
        ss << "\t\tFailure at index = " << index << ":\n";
    }
};

template <std::totally_ordered T, wrs::generic_allocator Allocator> struct IsPartitionError {
    using Type = IsPartitionErrorType;
    using IError = IsPartitionIndexError<T>;
    using allocator = std::allocator_traits<Allocator>::template rebind_alloc<IError>;

    IsPartitionError() : errorTypes(IS_PARTITION_ERROR_TYPE_NONE) {}
    IsPartitionError(Type type,
                     size_t heavyCount,
                     size_t lightCount,
                     size_t elementCount,
                     T pivot,
                     std::vector<IError, allocator> errors)
        : errorTypes(type), heavyCount(heavyCount), lightCount(lightCount),
          elementCount(elementCount), pivot(pivot), errors(std::move(errors)) {}

    IsPartitionErrorType errorTypes;
    size_t heavyCount;
    size_t lightCount;
    size_t elementCount;
    T pivot;
    std::vector<IError, allocator> errors;

    operator bool() const {
        return errorTypes != IS_PARTITION_ERROR_TYPE_NONE;
    }
    std::string message() const {
        std::stringstream ss;
        if (errorTypes & IS_PARTITION_ERROR_TYPE_INVALID_PARTITION_SIZES) {
            ss << "AssertionFailed: The size of the partition is incorrect:\n";
            ss << "\tExpected: " << elementCount << ", Got: " << heavyCount + lightCount << "\n";
            return ss.str();
        }

        ss << "AssertionFailed: \"prefix\" is not partition.\n";
        ss << "\tFailed at " << errors.size() << " out of " << elementCount << " elements\n";
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

template <std::totally_ordered T, wrs::generic_allocator Allocator = std::allocator<void>>
IsPartitionError<
    T,
    typename std::allocator_traits<Allocator>::template rebind_alloc<IsPartitionIndexError<T>>>
assert_is_partition(std::span<const T> heavy,
                    std::span<const T> light,
                    std::span<const T> elements,
                    T pivot,
                    const Allocator& alloc = {}) {
    using Error = IsPartitionError<
        T,
        typename std::allocator_traits<Allocator>::template rebind_alloc<IsPartitionIndexError<T>>>;
    using IError = Error::IError;
    using ErrorType = Error::Type;
    using ErrorAllocator = Error::allocator;
    using MapAllocator =
        std::allocator_traits<Allocator>::template rebind_alloc<std::pair<const T, size_t>>;
    using UsedMap =
        std::unordered_multimap<T, size_t, std::hash<T>, std::equal_to<T>, MapAllocator>;

    const size_t heavyCount = heavy.size();
    const size_t lightCount = light.size();
    const size_t elementCount = elements.size();

    if (heavyCount + lightCount != elementCount) {
        return Error(ErrorType::IS_PARTITION_ERROR_TYPE_INVALID_PARTITION_SIZES, heavyCount,
                     lightCount, elementCount, pivot, std::vector<IError, ErrorAllocator>{alloc});
    }

    UsedMap elementsUsed{MapAllocator(alloc)};
    elementsUsed.reserve(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        elementsUsed.insert(std::make_pair(elements[i], i));
    }

    std::vector<IError, ErrorAllocator> errors{ErrorAllocator(alloc)};
    errors.reserve(elementCount);

    ErrorType aggErr = IS_PARTITION_ERROR_TYPE_NONE;

    for (size_t i = 0; i < heavyCount; ++i) {
        const auto& h = heavy[i];
        IError err;
        err.index = i;
        err.shouldBeHeavy = h > pivot;
        err.type = IS_PARTITION_ERROR_TYPE_NONE;
        err.value = h;
        if (!err.shouldBeHeavy) {
            err.type = static_cast<ErrorType>(
                static_cast<unsigned int>(err.type) |
                static_cast<unsigned int>(IS_PARTITION_ERROR_TYPE_INVALID_PARTITION));
        }
        auto it = elementsUsed.find(h);
        bool found = it != elementsUsed.end();
        if (found) {
            err.elementIndex = it->second;
            elementsUsed.erase(it);
        } else {
            err.elementIndex = -1;
        }
        if (!found) {
            err.type = static_cast<ErrorType>(
                static_cast<unsigned int>(err.type) |
                static_cast<unsigned int>(IS_PARTITION_ERROR_TYPE_INVALID_ELEMENT));
        }
        if (err.type != IS_PARTITION_ERROR_TYPE_NONE) {
            errors.push_back(err);
            aggErr = static_cast<ErrorType>(static_cast<unsigned int>(aggErr) |
                                            static_cast<unsigned int>(err.type));
        }
    }
    for (size_t i = 0; i < lightCount; ++i) {
        const auto& l = light[i];
        IError err;
        err.index = i;
        err.shouldBeHeavy = l > pivot;
        err.type = IS_PARTITION_ERROR_TYPE_NONE;
        err.value = l;
        if (err.shouldBeHeavy) {
            err.type = static_cast<ErrorType>(
                static_cast<unsigned int>(err.type) |
                static_cast<unsigned int>(IS_PARTITION_ERROR_TYPE_INVALID_PARTITION));
        }
        auto it = elementsUsed.find(l);
        bool found = it != elementsUsed.end();
        if (found) {
            err.elementIndex = it->second;
            elementsUsed.erase(it);
        } else {
            err.elementIndex = -1;
        }
        if (!found) {
            err.type = static_cast<ErrorType>(
                static_cast<unsigned int>(err.type) |
                static_cast<unsigned int>(IS_PARTITION_ERROR_TYPE_INVALID_ELEMENT));
        }
        if (err.type != IS_PARTITION_ERROR_TYPE_NONE) {
            errors.push_back(err);
            aggErr = static_cast<ErrorType>(static_cast<unsigned int>(aggErr) |
                                            static_cast<unsigned int>(err.type));
        }
    }
    return Error(aggErr, heavyCount, lightCount, elementCount, pivot, std::move(errors));
}

namespace pmr {

template <std::totally_ordered T>
IsPartitionError<T, std::pmr::polymorphic_allocator<IsPartitionIndexError<T>>>
assert_is_partition(std::span<const T> heavy,
                    std::span<const T> light,
                    std::span<const T> elements,
                    T pivot,
                    const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::test::assert_is_partition<T, std::pmr::polymorphic_allocator<void>>(
        heavy, light, elements, pivot, alloc);
}

} // namespace pmr

} // namespace wrs::test
