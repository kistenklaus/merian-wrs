#pragma once

#include <array>
#include <cstddef>
#include <concepts>
#include <cstdint>
#include <glm/fwd.hpp>
#include <type_traits>

namespace wrs::layout::traits {
// Helper to detect arrays and get the underlying type
template <typename T>
concept IsStructLayout = requires { typename T::is_struct_layout_marker; };

template <typename T>
concept IsContiguousStructLayout = IsStructLayout<T> && T::contiguous;

template<typename S, typename Layout> 
concept CompatibleStructLayout = glsl::storage_qualified_struct<S> && 
  std::is_trivial_v<S> && 
  std::is_standard_layout_v<S> && 
  (S::storage_qualifier & Layout::storage);

template <typename... Attributes>
struct is_last_attribute_pointer {
    using last_type = typename std::tuple_element<sizeof...(Attributes) - 1, std::tuple<Attributes...>>::type::type;
    static constexpr bool value = std::is_pointer_v<last_type>;
};


} // namespace wrs::layout::traits
