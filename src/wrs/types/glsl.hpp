#pragma once

#include <cstdint>
#include <glm/fwd.hpp>
#include <type_traits>
namespace wrs::glsl {

enum class StorageQualifier : unsigned int {
    none = 0,
    std140 = 1 << 0,
    std430 = 1 << 1,
};

constexpr StorageQualifier operator|(StorageQualifier lhs, StorageQualifier rhs) {
    return static_cast<StorageQualifier>(static_cast<unsigned int>(lhs) | static_cast<unsigned int>(rhs));
}

constexpr bool operator&(StorageQualifier lhs, StorageQualifier rhs) {
    return static_cast<unsigned int>(lhs) & static_cast<unsigned int>(rhs);
}

using uint = uint32_t;
using float_t = float;
using sint = int32_t;

template <typename T>
concept vec2_like = std::is_same_v<T, glm::vec2>;

template <typename T>
concept vec3_like = std::is_same_v<T, glm::vec3>;

template <typename T>
concept vec4_like = std::is_same_v<T, glm::vec4>;


template <typename T>
concept scalar_like = std::is_same_v<T, wrs::glsl::float_t> || 
                         std::is_same_v<T, wrs::glsl::sint> || 
                         std::is_same_v<T, wrs::glsl::uint>;

template <typename T>
concept vec_like = vec2_like<T> || vec3_like<T> || vec4_like<T>;

template <typename T>
concept primitive_like = scalar_like<T> || vec_like<T>;

// Concepts for GLSL layout compatibility
template <typename T>
concept storage_qualified_struct = requires { 
{ T::storage_qualifier } -> std::convertible_to<wrs::glsl::StorageQualifier>;
};

}
