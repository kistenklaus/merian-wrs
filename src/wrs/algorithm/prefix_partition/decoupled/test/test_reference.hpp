#pragma once

#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_types.hpp"
#include "src/wrs/cpu/stable.hpp"
namespace wrs::test::decoupled_prefix_partition {


template<typename weight_t, typename Allocator>
void compute_reference(const std::vector<weight_t, Allocator> elements, weight_t pivot) {
  /* cpu::stable::partition(const std::vector<T> &weights, T pivot) */
}

}
