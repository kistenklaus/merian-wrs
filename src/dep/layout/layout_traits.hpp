#pragma once

#include "src/wrs/layout/StaticString.hpp"
#include "src/wrs/types/glsl.hpp"
#include <type_traits>

namespace wrs::layout::traits {

template <typename T>
concept IsPrimitiveLayout = requires { typename T::is_primitive_layout_marker; typename T::base_type;};

template <typename T>
concept IsArrayLayout = requires { 
  typename T::is_array_layout_marker; 
  typename T::base_type;
  /* typename T::storage; */
};

template<typename T>
concept IsPrimitiveArrayLayout = IsArrayLayout<T> && wrs::glsl::primitive_like<typename T::base_type>;


template<typename T>
concept IsComplexArrayLayout = IsArrayLayout<T> && !wrs::glsl::primitive_like<typename T::base_type>;

template <typename T>
concept IsStructLayout = requires { typename T::is_struct_layout_marker; };

template <typename T>
concept any_layout = IsPrimitiveLayout<T> || IsArrayLayout<T> || IsStructLayout<T>;

template <typename T>
concept IsSizedStructLayout = IsStructLayout<T> && T::sized;

template <typename T>
concept IsUnsizedStructLayout = IsStructLayout<T> && !T::sized;

template <typename T>
concept IsSizedLayout = IsSizedStructLayout<T> ||
  IsPrimitiveLayout<T>;

template<typename T>
concept IsUnsizedLayout = IsUnsizedStructLayout<T> ||
  IsArrayLayout<T>;


template<typename S, typename Layout> 
concept IsStorageCompatibleStruct = glsl::storage_qualified_struct<S> && 
  std::semiregular<S> && 
  std::is_standard_layout_v<S> && 
  (S::storage_qualifier & Layout::storage) && 
  (S::size(Layout::storage) == Layout::size()) && 
  (S::alignment(Layout::storage) == Layout::alignment());

template <typename... Attributes>
struct is_last_attribute_array {
    using last_type = typename std::tuple_element_t<sizeof...(Attributes) - 1, std::tuple<Attributes...>>::type;
    static constexpr bool value = std::is_pointer_v<last_type> || IsArrayLayout<last_type>;
};

template <typename... Attributes>
concept IsLastAttributeArray = requires {
    typename std::tuple_element_t<sizeof...(Attributes) - 1, std::tuple<Attributes...>>::type;
} && (
    std::is_pointer_v<typename std::tuple_element_t<sizeof...(Attributes) - 1, std::tuple<Attributes...>>::type> ||
    IsArrayLayout<typename std::tuple_element_t<sizeof...(Attributes) - 1, std::tuple<Attributes...>>::type>);

template <IsStructLayout Layout, StaticString AttributeName>
using struct_attribute_type = decltype(Layout{}.template get<AttributeName>());


} // namespace wrs::layout::traits
