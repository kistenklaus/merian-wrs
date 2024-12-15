#pragma once

#include "src/wrs/types/glsl.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/layout_traits.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

namespace wrs::layout {

template <std::size_t N> struct StaticString {
    char value[N];

    consteval StaticString(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }

    consteval operator std::string_view() const {
        return std::string_view(value, N - 1); // Exclude null terminator
    }
};

template <typename T, StaticString Name> struct Attribute {
    using type = T;
    static constexpr auto name = Name;
};

template <glsl::StorageQualifier Storage, typename... Attributes> class StructLayout {
  public:
    using is_struct_layout_marker = void;
    static constexpr glsl::StorageQualifier storage = Storage;
    static constexpr std::size_t ATTRIB_COUNT = sizeof...(Attributes);
    static constexpr bool contiguous = !wrs::layout::traits::is_last_attribute_pointer<Attributes...>::value;

    constexpr StructLayout(std::size_t offset = 0) : m_offset(offset) {}

    constexpr std::size_t offset() const {
      return m_offset;
    }

    template <StaticString Name> constexpr auto get() const {
        constexpr std::size_t index = findIndex<Name>();
        return get<index>();
    }

    constexpr vk::DeviceSize alignment() const {
        vk::DeviceSize max_alignment = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 using AttributeType =
                     typename std::tuple_element_t<Is, std::tuple<Attributes...>>::type;

                 if constexpr (wrs::glsl::primitive_like<AttributeType>) {
                     max_alignment = std::max(max_alignment,
                                              PrimitiveLayout<AttributeType, Storage>().alignment());
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
    constexpr vk::DeviceSize size() const requires(contiguous) {
        vk::DeviceSize total_size = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                using AttributeType = typename std::tuple_element_t<Is, std::tuple<Attributes...>>::type;

                if constexpr (wrs::glsl::primitive_like<AttributeType>) {
                    const auto layout = PrimitiveLayout<AttributeType, Storage>();
                    total_size = alignUp(total_size, layout.alignment()) + layout.size();
                } else if constexpr (wrs::layout::traits::IsStructLayout<AttributeType>) {
                    const auto layout = AttributeType{};
                    total_size = alignUp(total_size, layout.alignment()) + layout.size();
                }
            }()),
             ...);
        }(std::make_index_sequence<ATTRIB_COUNT>{});

        return alignUp(total_size, alignment());
    }
    constexpr vk::DeviceSize size(std::size_t arraySize) const requires(contiguous) {
        vk::DeviceSize total_size = 0;

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                using AttributeType = typename std::tuple_element_t<Is, std::tuple<Attributes...>>::type;

                if constexpr (wrs::glsl::primitive_like<AttributeType>) {
                    const auto layout = PrimitiveLayout<AttributeType, Storage>();
                    total_size = alignUp(total_size, layout.alignment()) + layout.size();
                } else if constexpr (std::is_pointer_v<AttributeType>) {
                    using BaseType = typename std::pointer_traits<AttributeType>::element_type;
                    const auto layout = ArrayLayout<BaseType, Storage>();
                    total_size = alignUp(total_size, layout.alignment()) + layout.size(arraySize);
                } else if constexpr (wrs::layout::traits::IsStructLayout<AttributeType>) {
                    const auto layout = AttributeType{};
                    total_size = alignUp(total_size, layout.alignment()) + layout.size();
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
          using BaseType = std::pointer_traits<AttributeType>::element_type;
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
                     constexpr AttributeLayout layout{};
                     offsets[Is] = alignUp(running_offset, layout.alignment());
                     running_offset = offsets[Is] + layout.size();
                 } else if constexpr (std::is_pointer<AttributeType>::value) {
                     using BaseType = std::pointer_traits<AttributeType>::element_type;
                     using AttributeLayout = ArrayLayout<BaseType, Storage>;
                     constexpr AttributeLayout layout;
                     offsets[Is] = alignUp(running_offset, layout.alignment());
                 } else if constexpr (wrs::layout::traits::IsStructLayout<AttributeType>) {
                     using AttributeLayout = AttributeType;
                     constexpr AttributeLayout layout;
                     if constexpr (requires {
                         layout.alignment(); 
                         layout.size(); 
                         }) {
                         offsets[Is] = alignUp(running_offset, layout.alignment());
                         running_offset = offsets[Is] + layout.size();
                     }
                 }
             }()),
             ...);
        }(std::make_index_sequence<ATTRIB_COUNT>{});

        return offsets;
    }
    // Align a size to the nearest multiple of alignment
    static constexpr vk::DeviceSize alignUp(vk::DeviceSize size, vk::DeviceSize alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    static constexpr std::array<vk::DeviceSize, ATTRIB_COUNT> m_offsets = computeOffsets();
    const vk::DeviceSize m_offset;
};

}; // namespace wrs::layout
