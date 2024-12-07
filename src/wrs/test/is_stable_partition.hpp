#pragma once

#include "src/wrs/test/is_partition.hpp"
#include <memory>
#include <span>
namespace wrs::test {

template <std::totally_ordered T, wrs::generic_allocator Allocator = std::allocator<void>>
IsPartitionError<
    T,
    typename std::allocator_traits<Allocator>::template rebind_alloc<IsPartitionIndexError<T>>>
assert_is_stable_partition(const std::span<T> heavy,
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
assert_is_stable_partition(const std::span<T> heavy,
                           const std::span<T> light,
                           const std::span<T> elements,
                           T pivot,
                           const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::test::assert_is_partition<T, std::pmr::polymorphic_allocator<void>>(
        heavy, light, elements, pivot, alloc);
}

} // namespace pmr

} // namespace wrs::test
