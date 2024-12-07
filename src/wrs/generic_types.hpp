#pragma once

#include "src/wrs/why.hpp"
#include <concepts>
#include <span>
#include <tuple>
#include <vector>
namespace wrs {

template <std::floating_point P> using alias_table_entry_t = std::tuple<P, std::size_t>;

template <wrs::arithmetic T> 
using split_t = std::tuple<std::size_t, std::size_t, T>;


template <wrs::arithmetic T, typename Allocator>
    requires(std::same_as<typename Allocator::value_type, T>)
using partition_t = std::tuple<std::span<T>, std::span<T>, std::vector<T, Allocator>>;
template <typename Allocator>
  requires (std::same_as<typename Allocator::value_type, std::size_t>)
using partition_indices_t =
    std::tuple<std::span<std::size_t>, std::span<std::size_t>, std::vector<std::size_t, Allocator>>;

} // namespace wrs
