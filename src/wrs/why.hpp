#pragma once

// This files contains a couple of easy helper functions or structs
// that for some reason don't exist in the stl yet.

#include <concepts>
#include <memory>
#include <type_traits>
namespace wrs {

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename Allocator>
concept generic_allocator = requires(Allocator alloc, std::size_t n) {
    typename Allocator::value_type;
    { alloc.allocate(n) } -> std::same_as<typename Allocator::value_type*>;
    { alloc.deallocate(static_cast<typename Allocator::value_type*>(nullptr), n) };
};

template <typename Allocator, typename V>
concept typed_allocator = generic_allocator<Allocator> &&
                          std::same_as<typename std::allocator_traits<Allocator>::value_type, V>;

// Computes ceil(a / b)
template <std::integral T>
T ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}


} // namespace wrs
