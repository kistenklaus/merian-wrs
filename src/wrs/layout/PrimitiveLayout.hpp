#pragma once

#include "src/wrs/types/glsl.hpp"
#include <cstddef>
namespace wrs::layout {

template <typename T, glsl::StorageQualifier Storage>
    requires wrs::glsl::primitive_like<T> 
class PrimitiveLayout {
  public:
    static constexpr glsl::StorageQualifier storage = Storage;
    using is_primitive_layout_marker = void;
    using base_type = T;
    constexpr PrimitiveLayout(std::size_t offset = 0) : m_offset(offset) {}
    // Compute size of the type based on the GLSL layout
    static constexpr std::size_t size() {
      return wrs::glsl::primitive_size<T, Storage>();
    }

    // Compute alignment of the type based on the GLSL layout
    static constexpr std::size_t alignment() {
      return wrs::glsl::primitive_alignment<T, Storage>();
    }

    // Access the base offset of the array
    [[nodiscard]] constexpr std::size_t offset() const {
        return m_offset;
    }

    void setMapped(void* mapped, T value) const {
        auto mappedBytes = static_cast<std::byte*>(mapped);
      T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
      *valueMapped = value;
    }

    T getFromMapped(void* mapped) const {
      auto* mappedBytes = static_cast<std::byte*>(mapped);
      T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
      return *valueMapped;
    }

    const std::size_t m_offset;
};

} // namespace wrs::layout
