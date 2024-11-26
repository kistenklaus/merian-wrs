#pragma once

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
namespace wrs::reference {

/**
 * Performs a reduction operation with the binary + operator!
 * Uses a reasonably numerically stable algorithm!
 *
 * We have implemented a hierarchical reduction algorithm, which 
 * guarantees that calls to the binary + operator is called only 
 * partial reductions of around the same amount of elements.
 * Compared to a linear scan reduction this reduces numerical
 * instabilities for large input sizes. 
 * (e.g avoids a + b = a if a >>> 0 and 0 < b ~ 0).
 *
 * Current implementation is unaware the the values of elements,
 * future implementation could utilize this to improve stability.
 * (e.g inital sorting).
 */
template<typename T, typename Allocator = std::allocator<T>>
T reduce(const std::span<T> elements, const Allocator& alloc = {}) {
  std::vector<T, Allocator> scratch(elements.begin(), elements.end(), alloc);
  for (size_t split = std::bit_ceil(elements.size()) >> 1; split > 0; split >>= 1) {
    for (size_t j = 0; j < split; j++) {
      if (j + split < scratch.size()) {
        scratch[j] = scratch[j] + scratch[j + split];
      }
    }
  }
  return scratch.front();
}


namespace pmr {

/**
 * Performs a reduction operation with the binary + operator!
 * Uses a reasonably numerically stable algorithm!
 *
 * We have implemented a hierarchical reduction algorithm, which 
 * guarantees that calls to the binary + operator is called only 
 * partial reductions of around the same amount of elements.
 * Compared to a linear scan reduction this reduces numerical
 * instabilities for large input sizes. 
 * (e.g avoids a + b = a if a >>> 0 and 0 < b ~ 0).
 *
 * Current implementation is unaware the the values of elements,
 * future implementation could utilize this to improve stability.
 * (e.g inital sorting).
 */
template<typename T>
T reduce(const std::span<T> elements, const std::pmr::polymorphic_allocator<T>& alloc = {}){
  return wrs::reference::reduce(elements, alloc);
}

}

}
