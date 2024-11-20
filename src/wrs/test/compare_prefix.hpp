#pragma once

#include <ranges>

namespace wrs::test {

enum class ComparePrefixErrorType {
  UNEQUAL_SIZE,
  UNEQUAL_SIMILAR_VALUES,
  UNEQUAL_UNSIMILAR_VALUES,
};

struct ComparePrefixResult {

};

template<std::ranges::sized_range ElementRange, std::ranges::sized_range PrefixRange>
ComparePrefixResult compare_prefix(const ElementRange& elements, const PrefixRange& prefix) {

}


}
