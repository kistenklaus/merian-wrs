#pragma once

#include "src/wrs/why.hpp"
#include "src/wrs/types/glsl.hpp"
#include "src/wrs/layout/layout_traits.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <span>
namespace wrs::layout {

template <typename T, glsl::StorageQualifier Storage> 
  requires (wrs::glsl::primitive_like<T> != //xor 
      wrs::layout::traits::IsContiguousStructLayout<T>)
  class ArrayLayout {
  public:
    static constexpr glsl::StorageQualifier storage = Storage;
    constexpr ArrayLayout(std::size_t offset = 0) : m_offset(offset) {}

    // Compute the size of the array based on the number of elements
    constexpr std::size_t size(std::size_t elementCount) const {
        return elementCount * alignUp(sizeof(T), alignment());
    }

    // Compute the alignment of the array
    constexpr std::size_t alignment() const {
        if constexpr (Storage == glsl::StorageQualifier::std140) {
            return std::max(sizeof(T),
                            static_cast<std::size_t>(16)); // std140 arrays align to 16 bytes
        } else {
            return alignof(T); // std430 uses the alignment of the element type
        }
    }

    constexpr std::size_t offset() const {
        return m_offset;
    }

    void setMapped(void* mapped, std::span<const T> value) const requires wrs::glsl::primitive_like<T> {
      std::byte* mappedBytes = reinterpret_cast<std::byte*>(mapped);
      T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
      std::memcpy(valueMapped, value.data(), value.size() * sizeof(T));
    }

    template<wrs::typed_allocator<T> Allocator = std::allocator<T>>
    std::vector<T, Allocator> getFromMapped(void* mapped, std::size_t N,
        const Allocator& alloc = {}) const requires wrs::glsl::primitive_like<T> {
      std::byte* mappedBytes = reinterpret_cast<std::byte*>(mapped);
      T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
      std::vector<T, Allocator> out{N, alloc};
      std::memcpy(out.data(), valueMapped, N * sizeof(T));
      return out;
    }

    std::pmr::vector<T> getPmrFromMapped(void* mapped, std::size_t N,
        const std::pmr::polymorphic_allocator<T>& alloc = {}) const requires wrs::glsl::primitive_like<T> {
      return get<std::pmr::polymorphic_allocator<T>>(mapped, N, alloc);
    }

    template<wrs::layout::traits::CompatibleStructLayout<T> S>
    void setMapped(void* mapped, std::span<const S> value) const requires wrs::layout::traits::IsContiguousStructLayout<T> {
      std::byte* mappedBytes = reinterpret_cast<std::byte*>(mapped);
      S* valueMapped = reinterpret_cast<S*>(mappedBytes + offset());
      std::memcpy(valueMapped, value.data(), value.size() * sizeof(S));
    }

    template<wrs::layout::traits::CompatibleStructLayout<T> S, wrs::typed_allocator<S> Allocator = std::allocator<S>>
    std::vector<S, Allocator> getFromMapped(void* mapped, std::size_t N,
        const Allocator& alloc = {}) const requires wrs::layout::traits::IsContiguousStructLayout<T> {
      std::vector<S, Allocator> out{N, alloc};
      std::byte* mappedBytes = reinterpret_cast<std::byte*>(mapped);
      S* valueMapped = reinterpret_cast<S*>(mappedBytes + offset());
      std::memcpy(out.data(), valueMapped, N * sizeof(S));
    }

    auto get(std::size_t i) {
      if constexpr (wrs::glsl::primitive_like<T>) {
        constexpr auto element_alignment = []() constexpr {
            if constexpr (Storage == glsl::StorageQualifier::std140) {
                return std::max(alignof(T), std::size_t(16)); // std140 arrays align to 16 bytes
            } else {
                return alignof(T); // std430 uses natural alignment
            }
        }();

        constexpr auto element_size = []() constexpr {
            if constexpr (Storage == glsl::StorageQualifier::std140) {
                return std::max(sizeof(T), std::size_t(16)); // std140 arrays pad elements to 16 bytes
            } else {
                return sizeof(T); // std430 uses natural size
            }
        }();

        return PrimitiveLayout<T, Storage>(offset() + i * element_size);
      } else  { // assert wrs::layout::traits::IsContiguousStructLayout<T>
        return T{offset() + i * T{}.size()};
      }
    }


  public:
    // Helper function to align sizes
    static constexpr std::size_t alignUp(std::size_t size, std::size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    const std::size_t m_offset;
};

} // namespace wrs::layout
