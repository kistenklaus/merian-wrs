#pragma once

#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/StaticString.hpp"
#include "src/wrs/layout/layout_traits.hpp"
#include "src/wrs/types/glsl.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>

namespace wrs::layout {

template <glsl::StorageQualifier Storage, typename... Attributes> class StructLayout {
  public:
    using is_struct_layout_marker = void;
    static constexpr glsl::StorageQualifier storage = Storage;
    static constexpr std::size_t ATTRIB_COUNT = sizeof...(Attributes);
    static constexpr bool sized = !wrs::layout::traits::IsLastAttributeArray<Attributes...>;

    constexpr StructLayout(const std::size_t offset = 0) : m_offset(offset) {}

    [[nodiscard]] constexpr std::size_t offset() const {
        return m_offset;
    }

    template <StaticString Name> constexpr auto get() const {
        constexpr std::size_t index = findIndex<Name>();
        return get<index>();
    }

    template <wrs::layout::traits::IsStorageCompatibleStruct<StructLayout> S>
    void setMapped(void* mapped, const S& s) const
        requires wrs::layout::traits::IsSizedStructLayout<StructLayout>
    {
        auto* mappedBytes = static_cast<std::byte*>(mapped);
        S* valueMapped = reinterpret_cast<S*>(mappedBytes + offset());
        std::memcpy(valueMapped, &s, sizeof(S));
    }

    template <wrs::layout::traits::IsStorageCompatibleStruct<StructLayout> S>
    S getFromMapped(void* mapped) const
        requires wrs::layout::traits::IsSizedStructLayout<StructLayout>
    {
        auto mappedBytes = static_cast<std::byte*>(mapped);
        S out;
        const S* valueMapped = reinterpret_cast<S*>(mappedBytes + offset());
        std::memcpy(&out, valueMapped, sizeof(S));
        return out;
    }

    static constexpr vk::DeviceSize alignment() {
        vk::DeviceSize max_alignment = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 using AttributeType =
                     typename std::tuple_element_t<Is, std::tuple<Attributes...>>::type;

                 if constexpr (wrs::glsl::primitive_like<AttributeType>) {
                     max_alignment = std::max(
                         max_alignment, PrimitiveLayout<AttributeType, Storage>().alignment());
                 } else if constexpr (std::is_pointer_v<AttributeType>) {
                     using BaseType = typename std::pointer_traits<AttributeType>::element_type;
                     max_alignment =
                         std::max(max_alignment, ArrayLayout<BaseType, Storage>().alignment());
                 } else if constexpr (wrs::layout::traits::IsStructLayout<AttributeType>) {
                     max_alignment = std::max(max_alignment, AttributeType{}.alignment());
                 }
             }()),
             ...);
        }(std::make_index_sequence<ATTRIB_COUNT>{});

        return max_alignment;
    }
    static constexpr vk::DeviceSize size()
        requires(wrs::layout::traits::IsSizedStructLayout<StructLayout>)
    {
        vk::DeviceSize total_size = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 using AttributeType =
                     typename std::tuple_element_t<Is, std::tuple<Attributes...>>::type;

                 if constexpr (wrs::glsl::primitive_like<AttributeType>) {
                     using Layout = PrimitiveLayout<AttributeType, Storage>;
                     total_size = alignUp(total_size, Layout::alignment()) + Layout::size();
                 } else if constexpr (wrs::layout::traits::IsStructLayout<AttributeType>) {
                     using Layout = AttributeType;
                     total_size = alignUp(total_size, Layout::alignment()) + Layout::size();
                 }
             }()),
             ...);
        }(std::make_index_sequence<ATTRIB_COUNT>{});

        return alignUp(total_size, alignment());
    }
    static constexpr vk::DeviceSize size(std::size_t arraySize)
        requires(wrs::layout::traits::IsUnsizedStructLayout<StructLayout>)
    {
        vk::DeviceSize total_size = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 using AttributeType =
                     typename std::tuple_element_t<Is, std::tuple<Attributes...>>::type;

                 if constexpr (wrs::glsl::primitive_like<AttributeType>) {
                     using Layout = PrimitiveLayout<AttributeType, Storage>;
                     total_size = alignUp(total_size, Layout::alignment()) + Layout::size();
                 } else if constexpr (std::is_pointer_v<AttributeType>) {
                     using BaseType = typename std::pointer_traits<AttributeType>::element_type;
                     using Layout = ArrayLayout<BaseType, Storage>;
                     total_size =
                         alignUp(total_size, Layout::alignment()) + Layout::size(arraySize);
                 } else if constexpr (wrs::layout::traits::IsArrayLayout<AttributeType>) {
                     using Layout = AttributeType;
                     total_size =
                         alignUp(total_size, Layout::alignment()) + Layout::size(arraySize);
                 } else if constexpr (wrs::layout::traits::IsSizedStructLayout<AttributeType>) {
                     using Layout = AttributeType;
                     total_size = alignUp(total_size, Layout::alignment()) + Layout::size();
                 }
                 if constexpr (wrs::layout::traits::IsUnsizedStructLayout<AttributeType>) {
                     using Layout = AttributeType;
                     total_size =
                         alignUp(total_size, Layout::alignment()) + Layout::size(arraySize);
                 }
             }()),
             ...);
        }(std::make_index_sequence<ATTRIB_COUNT>{});

        return alignUp(total_size, alignment());
    }

  private:
    // Getter to retrieve the PrimitiveLayout for a given index
    template <std::size_t Index> constexpr auto get() const {
        static_assert(Index < ATTRIB_COUNT, "Index out of range!");
        using AttributeType = typename std::tuple_element_t<Index, std::tuple<Attributes...>>::type;

        if constexpr (wrs::glsl::primitive_like<AttributeType>) {
            return PrimitiveLayout<AttributeType, Storage>(m_offset + m_offsets[Index]);
        } else if constexpr (std::is_pointer_v<AttributeType>) {
            using BaseType = typename std::pointer_traits<AttributeType>::element_type;
            return ArrayLayout<BaseType, Storage>(m_offset + m_offsets[Index]);
        } else if constexpr (wrs::layout::traits::IsStructLayout<AttributeType>) {
            return AttributeType(m_offset + m_offsets[Index]);
        } else {
            return AttributeType{}; // Should never happen
        }
    }
    template <StaticString Name, std::size_t Index = 0> static consteval std::size_t findIndex() {
        if constexpr (Index >= ATTRIB_COUNT) {
            throw std::runtime_error("Attribute not found");
        } else if constexpr (std::string_view(Name) ==
                             std::tuple_element_t<Index, std::tuple<Attributes...>>::name) {
            return Index;
        } else {
            return findIndex<Name, Index + 1>();
        }
    }

    template <std::size_t Index> static consteval vk::DeviceSize sizeOfElement() {
        if constexpr (Index < sizeof...(Attributes)) {
            return sizeof(typename std::tuple_element_t<Index, std::tuple<Attributes...>>::type);
        } else {
            return 0; // Should never happen
        }
    }

    static consteval auto computeOffsets() {
        std::array<vk::DeviceSize, ATTRIB_COUNT> offsets{};
        vk::DeviceSize running_offset = 0;

        // Use std::index_sequence for compile-time iteration
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            // Use fold expression to iterate over the indices
            (([&]() {
                 using AttributeType =
                     typename std::tuple_element_t<Is, std::tuple<Attributes...>>::type;
                 if constexpr (wrs::glsl::primitive_like<AttributeType>) {
                     using AttributeLayout = PrimitiveLayout<AttributeType, Storage>;
                     offsets[Is] = alignUp(running_offset, AttributeLayout::alignment());
                     running_offset = offsets[Is] + AttributeLayout::size();
                 } else if constexpr (std::is_pointer_v<AttributeType>) {
                     using BaseType = typename std::pointer_traits<AttributeType>::element_type;
                     using AttributeLayout = ArrayLayout<BaseType, Storage>;
                     offsets[Is] = alignUp(running_offset, AttributeLayout::alignment());
                 } else if constexpr (traits::IsArrayLayout<AttributeType>) {
                     using AttributeLayout = AttributeType;
                     offsets[Is] = alignUp(running_offset, AttributeLayout::alignment());
                 } else if constexpr (wrs::layout::traits::IsSizedStructLayout<AttributeType>) {
                     using AttributeLayout = AttributeType;
                     offsets[Is] = alignUp(running_offset, AttributeLayout::alignment());
                     running_offset = offsets[Is] + AttributeLayout::size();
                 } else if constexpr (wrs::layout::traits::IsUnsizedStructLayout<AttributeType>) {
                     using AttributeLayout = AttributeType;
                     offsets[Is] = alignUp(running_offset, AttributeLayout::alignment());
                 }
             }()),
             ...);
        }(std::make_index_sequence<ATTRIB_COUNT>{});

        return offsets;
    }
    // Align a size to the nearest multiple of alignment
    static constexpr vk::DeviceSize alignUp(const vk::DeviceSize size,
                                            const vk::DeviceSize alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    static constexpr std::array<vk::DeviceSize, ATTRIB_COUNT> m_offsets = computeOffsets();
    const vk::DeviceSize m_offset;
};


}; // namespace wrs::layout
