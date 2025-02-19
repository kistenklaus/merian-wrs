#pragma once

#include "src/host/reference/reduce.hpp"
#include "src/host/why.hpp"
#include <span>
namespace host::reference {

template <arithmetic T, typed_allocator<T> Allocator = std::allocator<T>>
T mean(const std::span<T> elements, const Allocator& alloc = {}) {
    // NOTE: probably not the most numerically stable solution.
    T reduction = reference::reduce<T, Allocator>(elements, alloc);
    return reduction / static_cast<T>(elements.size());
}

namespace pmr {
template <arithmetic T>
T mean(const std::span<T> elements, const std::pmr::polymorphic_allocator<T> alloc = {}) {
    return reference::mean<T, pmr_alloc<T>>(elements, alloc);
}
} // namespace pmr

} // namespace host::reference
