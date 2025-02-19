#pragma once

#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/layout_traits.hpp"
#include "src/wrs/types/glsl.hpp"
#include "src/wrs/why.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <span>
#include <vector>
namespace wrs::layout {

template <typename T, glsl::StorageQualifier Storage>
    requires((wrs::glsl::primitive_like<T> || traits::IsPrimitiveLayout<T>) != // xor
             wrs::layout::traits::IsSizedStructLayout<T>)
class ArrayLayout {
  public:
    using is_array_layout_marker = void;
    using base_type = T;
    static constexpr glsl::StorageQualifier storage = Storage;
    constexpr ArrayLayout(const std::size_t offset = 0) : m_offset(offset) {}

    static constexpr std::size_t size(std::size_t elementCount)
        requires wrs::glsl::primitive_like<T>
    {
        return elementCount * alignUp(glsl::primitive_size<T, Storage>(), alignment());
    }

    static constexpr std::size_t size(std::size_t elementCount)
        requires wrs::layout::traits::IsSizedStructLayout<T>
    {
        return elementCount * alignUp(T::size(), alignment());
    }

    static constexpr std::size_t alignment()
        requires wrs::glsl::primitive_like<T>
    {
        if constexpr (Storage == glsl::StorageQualifier::std140) {
            return std::max(glsl::primitive_alignment<T, Storage>(),
                            static_cast<std::size_t>(16)); // std140 arrays align to 16 bytes
        } else {
            return alignof(T); // std430 uses the alignment of the element type
        }
    }

    static constexpr std::size_t alignment()
        requires wrs::layout::traits::IsSizedStructLayout<T>
    {
        if constexpr (Storage == glsl::StorageQualifier::std140) {
            return std::max(T::alignment(),
                            static_cast<std::size_t>(16)); // std140 arrays align to 16 bytes
        } else {
            return T::alignment();
        }
    }

    [[nodiscard]] constexpr std::size_t offset() const {
        return m_offset;
    }

    void setMapped(void* mapped, std::span<const T> value) const
        requires wrs::glsl::primitive_like<T>
    {
        auto* mappedBytes = static_cast<std::byte*>(mapped);
        T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
        if constexpr (glsl::has_contiguous_primitive_array_storage<T, Storage>()) {
            std::memcpy(valueMapped, value.data(), value.size() * sizeof(T));
        } else {
            auto* head = reinterpret_cast<std::byte*>(valueMapped);
            const std::size_t stride = alignUp(glsl::primitive_size<T, Storage>(), alignment());
            for (std::size_t i = 0; i < value.size(); ++i) {
                T* entryMapped = reinterpret_cast<T*>(head);
                *entryMapped = value[i];
                head += stride;
            }
        }
    }

    template <wrs::typed_allocator<T> Allocator = std::allocator<T>>
    std::vector<T, Allocator>
    getFromMapped(void* mapped, std::size_t N, const Allocator& alloc = {}) const
        requires wrs::glsl::primitive_like<T>
    {
        auto* mappedBytes = static_cast<std::byte*>(mapped);
        T* valueMapped = reinterpret_cast<T*>(mappedBytes + offset());
        std::vector<T, Allocator> out{N, alloc};
        if constexpr (glsl::has_contiguous_primitive_array_storage<T, Storage>()) {
            std::memcpy(out.data(), valueMapped, N * sizeof(T));
        } else {
            auto* head = reinterpret_cast<std::byte*>(valueMapped);
            const std::size_t stride = alignUp(glsl::primitive_size<T, Storage>(), alignment());
            for (std::size_t i = 0; i < N; ++i) {
                T* entryMapped = reinterpret_cast<T*>(head);
                out[i] = *entryMapped;
                head += stride;
            }
        }
        return out;
    }

    template <wrs::layout::traits::IsStorageCompatibleStruct<T> S>
    void setMapped(void* mapped, std::span<const S> value) const
        requires wrs::layout::traits::IsSizedStructLayout<T>
    {
        auto* mappedBytes = static_cast<std::byte*>(mapped);
        S* valueMapped = reinterpret_cast<S*>(mappedBytes + offset());
        std::memcpy(valueMapped, value.data(), value.size() * sizeof(S));
    }

    template <wrs::layout::traits::IsStorageCompatibleStruct<T> S,
              wrs::typed_allocator<S> Allocator = std::allocator<S>>
    std::vector<S, Allocator>
    getFromMapped(void* mapped, std::size_t N, const Allocator& alloc = {}) const
        requires wrs::layout::traits::IsSizedStructLayout<T>
    {
        std::vector<S, Allocator> out{N, alloc};
        auto* mappedBytes = static_cast<std::byte*>(mapped);
        const S* valueMapped = reinterpret_cast<S*>(mappedBytes + offset());
        std::memcpy(out.data(), valueMapped, N * sizeof(S));
        return out;
    }

    auto get(std::size_t i) {
        if constexpr (wrs::glsl::primitive_like<T>) {
            constexpr auto element_alignment = []() constexpr {
                if constexpr (Storage == glsl::StorageQualifier::std140) {
                    return std::max(alignof(T), static_cast<std::size_t>(
                                                    16)); // std140 arrays align to 16 bytes
                } else {
                    return alignof(T); // std430 uses natural alignment
                }
            }();

            constexpr auto element_size = []() constexpr {
                if constexpr (Storage == glsl::StorageQualifier::std140) {
                    return std::max(sizeof(T),
                                    static_cast<std::size_t>(16)); // std140 arrays pad elements to 16 bytes
                } else {
                    return sizeof(T); // std430 uses natural size
                }
            }();

            return PrimitiveLayout<T, Storage>(offset() + i * element_size);
        } else { // assert wrs::layout::traits::IsContiguousStructLayout<T>
            return T{offset() + i * T::size()};
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
