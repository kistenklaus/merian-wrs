#pragma once

#include <concepts>
#include <cstdint>
#include <glm/fwd.hpp>
#include <type_traits>
namespace wrs::glsl {

enum class StorageQualifier : unsigned int {
    std140 = 1 << 0,
    std430 = 1 << 1,
};

constexpr StorageQualifier operator|(StorageQualifier lhs, StorageQualifier rhs) {
    return static_cast<StorageQualifier>(static_cast<unsigned int>(lhs) |
                                         static_cast<unsigned int>(rhs));
}

constexpr bool operator&(StorageQualifier lhs, StorageQualifier rhs) {
    return static_cast<unsigned int>(lhs) & static_cast<unsigned int>(rhs);
}

using uint = uint32_t;
using uint64 = uint64_t;
using sint64 = int64_t;
using f32 = float;
using sint = int32_t;

template <typename T>
concept vec2_like = std::is_same_v<T, glm::vec2>;

template <typename T>
concept vec3_like = std::is_same_v<T, glm::vec3>;

template <typename T>
concept vec4_like = std::is_same_v<T, glm::vec4>;

template <typename T>
concept scalar_like = std::is_same_v<T, wrs::glsl::f32> || std::is_same_v<T, wrs::glsl::sint> ||
                      std::is_same_v<T, wrs::glsl::uint>;
template <typename T>
concept int64_like = std::is_same_v<T, wrs::glsl::uint64> || std::is_same_v<T, wrs::glsl::sint64>;

template <typename T>
concept vec_like = vec2_like<T> || vec3_like<T> || vec4_like<T>;

template <typename T>
concept primitive_like = scalar_like<T> || vec_like<T> || int64_like<T>;

template <scalar_like T, StorageQualifier Storage> constexpr std::size_t primitive_alignment() {
    return 4;
}
template <scalar_like T, StorageQualifier Storage> constexpr std::size_t primitive_size() {
    return 4;
}

template <scalar_like T> constexpr std::size_t primitive_alignment(glsl::StorageQualifier) {
    return 4;
}
template <scalar_like T> constexpr std::size_t primitive_size(glsl::StorageQualifier) {
    return 4;
}


template <scalar_like T, StorageQualifier Storage>
constexpr bool has_contiguous_primitive_array_storage() {
    switch (Storage) {
    case StorageQualifier::std140:
        return false;
    case StorageQualifier::std430:
        return true;
    }
}

template <int64_like T, StorageQualifier Storage> constexpr std::size_t primitive_alignment() {
    return 8;
}
template <int64_like T, StorageQualifier Storage> constexpr std::size_t primitive_size() {
    return 8;
}

template <int64_like T> constexpr std::size_t primitive_alignment(glsl::StorageQualifier) {
    return 8;
}
template <int64_like T> constexpr std::size_t primitive_size(glsl::StorageQualifier) {
    return 8;
}

template <int64_like T, StorageQualifier Storage>
constexpr bool has_contiguous_primitive_array_storage() {
    switch (Storage) {
    case StorageQualifier::std140:
        return false;
    case StorageQualifier::std430:
        return true;
    }
}

template <vec2_like T, StorageQualifier Storage> constexpr std::size_t primitive_alignment() {
    return 8;
}
template <vec2_like T, StorageQualifier Storage> constexpr std::size_t primitive_size() {
    return 8;
}

template <vec2_like T> constexpr std::size_t primitive_alignment(glsl::StorageQualifier) {
    return 8;
}
template <vec2_like T> constexpr std::size_t primitive_size(glsl::StorageQualifier) {
    return 8;
}

template <vec2_like T, StorageQualifier Storage>
constexpr bool has_contiguous_primitive_array_storage() {
    switch (Storage) {
    case StorageQualifier::std140:
        return false;
    case StorageQualifier::std430:
        return true;
    }
}

template <vec3_like T, StorageQualifier Storage> constexpr std::size_t primitive_alignment() {
    return 16;
}
template <vec3_like T, StorageQualifier Storage> constexpr std::size_t primitive_size() {
    return 12;
}

template <vec3_like T> constexpr std::size_t primitive_alignment(glsl::StorageQualifier) {
    return 16;
}
template <vec3_like T> constexpr std::size_t primitive_size(glsl::StorageQualifier) {
    return 12;
}

template <vec3_like T, StorageQualifier Storage>
constexpr bool has_contiguous_primitive_array_storage() {
    switch (Storage) {
    case StorageQualifier::std140:
        return false;
    case StorageQualifier::std430:
        return false;
    }
}

template <vec4_like T, StorageQualifier Storage> constexpr std::size_t primitive_alignment() {
    return 16;
}
template <vec4_like T, StorageQualifier Storage> constexpr std::size_t primitive_size() {
    return 16;
}

template <vec4_like T> constexpr std::size_t primitive_alignment(glsl::StorageQualifier) {
    return 16;
}
template <vec4_like T> constexpr std::size_t primitive_size(glsl::StorageQualifier) {
    return 16;
}

template <vec4_like T, StorageQualifier Storage>
constexpr bool has_contiguous_primitive_array_storage() {
    switch (Storage) {
    case StorageQualifier::std140:
        return true;
    case StorageQualifier::std430:
        return true;
    }
}

// Concepts for GLSL layout compatibility
template <typename T>
concept storage_qualified_struct = requires(T s, StorageQualifier storage) {
    { T::storage_qualifier };
    { T::alignment(storage) } -> std::convertible_to<std::size_t>;
    { T::size(storage) } -> std::convertible_to<std::size_t>;
};

} // namespace wrs::glsl
