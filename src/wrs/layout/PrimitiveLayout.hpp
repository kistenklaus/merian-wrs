#pragma once

#include "src/wrs/layout/layout_traits.hpp"
#include "src/wrs/types/glsl.hpp"
#include <cstddef>
#include <glm/fwd.hpp>
namespace wrs::layout {

template <typename T, glsl::StorageQualifier Storage>
    requires wrs::glsl::primitive_like<T> || wrs::glsl::vec2_like<T> || wrs::glsl::vec3_like<T> ||
             wrs::glsl::vec4_like<T>
class PrimitiveLayout {
  public:
    static constexpr glsl::StorageQualifier storage = Storage;
    constexpr PrimitiveLayout(std::size_t offset = 0) : m_offset(offset) {}
    // Compute size of the type based on the GLSL layout
    constexpr std::size_t size() const {
        if constexpr (Storage == glsl::StorageQualifier::std140) {
            return std140Size();
        } else {
            return std430Size();
        }
    }

    // Compute alignment of the type based on the GLSL layout
    constexpr std::size_t alignment() const {
        if constexpr (Storage == glsl::StorageQualifier::std140) {
            return std140Alignment();
        } else {
            return std430Alignment();
        }
    }

    // Access the base offset of the array
    constexpr std::size_t offset() const {
        return m_offset;
    }

    void set(void* mapped, T value) const {
      std::byte* mappedBytes = reinterpret_cast<std::byte*>(mapped);
      T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
      *valueMapped = value;
    }

    T get(void* mapped) const {
      std::byte* mappedBytes = reinterpret_cast<std::byte*>(mapped);
      T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
      return *valueMapped;
    }

  public:
    // GLSL std140 size rules
    consteval std::size_t std140Size() const {
        if constexpr (wrs::glsl::scalar_like<T>) {
            return sizeof(T); // Scalars
        } else if constexpr (wrs::glsl::vec2_like<T>) {
            return 8; // vec2
        } else if constexpr (wrs::glsl::vec3_like<T> || wrs::glsl::vec4_like<T>) {
            return 16; // vec3 is padded to 16 bytes, vec4 naturally fits
        }
        return sizeof(T); // Default fallback for unknown types
    }

    // GLSL std140 alignment rules
    consteval std::size_t std140Alignment() const {
        if constexpr (wrs::glsl::scalar_like<T>) {
            return alignof(T); // Scalars
        } else if constexpr (wrs::glsl::vec2_like<T>) {
            return 8; // vec2
        } else if constexpr (wrs::glsl::vec3_like<T> || wrs::glsl::vec4_like<T>) {
            return 16; // vec3 and vec4 are aligned to 16 bytes
        }
        return alignof(T); // Default fallback for unknown types
    }

    // GLSL std430 size rules
    consteval std::size_t std430Size() const {
        return sizeof(T); // Scalars and vectors match their native C++ sizes
    }

    // GLSL std430 alignment rules
    consteval std::size_t std430Alignment() const {
        if constexpr (wrs::glsl::vec3_like<T> || wrs::glsl::vec4_like<T>) {
            return 16; // vec3 and vec4 align to 16 bytes
        }
        return alignof(T); // Scalars and vectors match native alignments
    }
    const std::size_t m_offset;
};

} // namespace wrs::layout
