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

enum IsPartitionErrorType {
    IS_PARTITION_ERROR_TYPE_NONE = 0,
    IS_PARTITION_ERROR_TYPE_INVALID_PARTITION_SIZES = 1,
    IS_PARTITION_ERROR_TYPE_INVALID_PARTITION = 2,
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

    size_t heavyErrors = 0;
    size_t lightErrors = 0;
    std::vector<bool, BAllocator> elementUsed(elementCount, BAllocator(alloc));
    std::vector<IError, ErrorAllocator> errors{ErrorAllocator(alloc)};
    errors.reserve(elements.size());
    for (size_t i = 0; i < heavyCount; ++i) {
        const auto& h = heavy.at(i);
        IError err;
        err.index = i;
        if (h <= pivot) {

            /* IsPartitionErrorType type; */
            /* size_t index; */
            /* bool shouldBeHeavy; */
            /* bool isHeavy; */
            /* T value; */
        }
        bool found = false;
        for (size_t j = 0; j < elementCount; ++j) {
            if (elementUsed[j] == false && elements[j] == h) {
                elementUsed[j] = true;
                found = true;
                break;
            }
        }
        if (!found) {
        }
    }
    for (const auto& l : light) {
        if (l > pivot) {
            lightErrors += 1;
        }
    }
}

} // namespace wrs::test
