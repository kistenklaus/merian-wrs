#pragma once

#include <algorithm>
#include <compare>
#include <memory>
#include <ranges>
#include <set>
#include <span>
#include <unordered_set>
#include <vector>
namespace wrs::test {

namespace concepts {

template <typename T>
concept partially_ordered = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
    { a <= b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
    { a >= b } -> std::convertible_to<bool>;
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
};

} // namespace concepts

enum IsPartitionErrorType : unsigned int{
    IS_PARTITION_ERROR_TYPE_NONE = 0,
    IS_PARTITION_ERROR_TYPE_INVALID_PARTITION_SIZES = 1,
    IS_PARTITION_ERROR_TYPE_INVALID_PARTITION = 2, // element assigned to the wrong partition
    IS_PARTITION_ERROR_TYPE_INVALID_ELEMENT = 4,
};

template <typename T> struct IsPartitionIndexError {
    IsPartitionErrorType type;
    size_t index;
    size_t elementIndex;
    bool shouldBeHeavy;
    bool isHeavy;
    T value;
};

template <typename T, typename Allocator> struct IsPartitionError {
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

    operator bool() {
        return errorTypes != IS_PARTITION_ERROR_TYPE_NONE;
    }
};

template <concepts::partially_ordered T, typename Allocator = std::allocator<void>>
IsPartitionError<
    T,
    typename std::allocator_traits<Allocator>::template rebind_alloc<IsPartitionIndexError<T>>>
assert_is_partition(const std::span<T> heavy,
                    const std::span<T> light,
                    const std::span<T> elements,
                    T pivot,
                    const Allocator& alloc = {}) {
    using Error = IsPartitionError<
        T,
        typename std::allocator_traits<Allocator>::template rebind_alloc<IsPartitionIndexError<T>>>;
    using IError = Error::IError;
    using ErrorType = Error::Type;
    using ErrorAllocator = Error::allocator;
    using BAllocator = std::allocator_traits<Allocator>::template rebind_alloc<bool>;

    const size_t heavyCount = heavy.size();
    const size_t lightCount = light.size();
    const size_t elementCount = elements.size();

    if (heavyCount + lightCount != elementCount) {
        return Error(ErrorType::IS_PARTITION_ERROR_TYPE_INVALID_PARTITION_SIZES, heavyCount,
                     lightCount, elementCount, pivot, std::vector<IError, ErrorAllocator>{alloc});
    }

    std::vector<bool, BAllocator> elementUsed(elementCount, BAllocator(alloc));
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
        bool found = false;
        size_t j = 0;
        for (; j < elementCount; ++j) {
            if (elementUsed[j] == false && elements[j] == h) {
                elementUsed[j] = true;
                found = true;
                break;
            }
        }
        err.elementIndex = j;
        if (!found) {
            err.type = static_cast<ErrorType>(
                static_cast<unsigned int>(err.type) |
                static_cast<unsigned int>(IS_PARTITION_ERROR_TYPE_INVALID_ELEMENT));
        }
        if (err.type != IS_PARTITION_ERROR_TYPE_NONE) {
            errors.push_back(err);
            aggErr = static_cast<ErrorType>(
                static_cast<unsigned int>(aggErr) |
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
        bool found = false;
        size_t j = 0;
        for (; j < elementCount; ++j) {
            if (elementUsed[j] == false && elements[j] == l) {
                elementUsed[j] = true;
                found = true;
                break;
            }
        }
        err.elementIndex = j;
        if (!found) {
            err.type = static_cast<ErrorType>(
                static_cast<unsigned int>(err.type) |
                static_cast<unsigned int>(IS_PARTITION_ERROR_TYPE_INVALID_ELEMENT));
        }
        if (err.type != IS_PARTITION_ERROR_TYPE_NONE) {
            errors.push_back(err);
            aggErr = static_cast<ErrorType>(
                static_cast<unsigned int>(aggErr) |
                static_cast<unsigned int>(err.type));
        }
    }
    return Error(aggErr, heavyCount, lightCount, elementCount, pivot, std::move(errors));
}

namespace pmr {

template <concepts::partially_ordered T>
IsPartitionError<T, std::pmr::polymorphic_allocator<IsPartitionIndexError<T>>>
assert_is_partition(const std::span<T> heavy,
                    const std::span<T> light,
                    const std::span<T> elements,
                    T pivot,
                    const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::test::assert_is_partition<T, std::pmr::polymorphic_allocator<void>>(heavy, light, elements,
                                                                             pivot, alloc);
}

} // namespace pmr

} // namespace wrs::test
