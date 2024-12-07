#pragma once

#include "src/wrs/reference/reduce.hpp"
#include <span>
namespace wrs::reference {

template <typename T, typename Allocator = std::allocator<T>>
T mean(const std::span<T> elements, const Allocator& alloc = {}) {
    // NOTE: probably not the most numerically stable solution. 
    T reduction = wrs::reference::tree_reduction(elements, alloc);
    return reduction / static_cast<T>(elements.size());
}


namespace pmr {
template<typename T>
T mean(const std::span<T> elements, const std::pmr::polymorphic_allocator<T> alloc = {}) {
  return wrs::reference::mean(elements, alloc);
}
}

} // namespace wrs::reference
