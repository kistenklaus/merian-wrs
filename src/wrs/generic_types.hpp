#pragma once

#include "src/wrs/why.hpp"
#include <concepts>
#include <span>
#include <tuple>
#include <vector>
namespace wrs {

template <std::floating_point P, std::integral I> using alias_table_entry_t = std::tuple<P, I>;

template <std::floating_point P,
          std::integral I,
          wrs::typed_allocator<wrs::alias_table_entry_t<P, I>> Allocator = std::allocator<wrs::alias_table_entry_t<P, I>>>
using alias_table_t = std::vector<wrs::alias_table_entry_t<P, I>, Allocator>;

template <std::floating_point P,
          std::integral I>
using const_alias_table_ref = std::span<const wrs::alias_table_entry_t<P, I>>;

template <wrs::arithmetic T, std::integral I> using split_t = std::tuple<I, I, T>;

template <wrs::arithmetic T, wrs::typed_allocator<T> Allocator>
using partition_t = std::tuple<std::span<T>, std::span<T>, std::vector<T, Allocator>>;
template <std::integral I, wrs::typed_allocator<I> Allocator>
using partition_indices_t = std::tuple<std::span<I>, std::span<I>, std::vector<I, Allocator>>;

namespace pmr {

template <wrs::arithmetic T>
using partition_t = wrs::partition_t<T, std::pmr::polymorphic_allocator<T>>;

template <std::integral I>
using partition_indices_t = wrs::partition_indices_t<I, std::pmr::polymorphic_allocator<I>>;

template <std::floating_point P, std::integral I>
using alias_table_t =
    wrs::alias_table_t<P, I, std::pmr::polymorphic_allocator<wrs::alias_table_entry_t<P, I>>>;

} // namespace pmr

} // namespace wrs
