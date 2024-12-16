#pragma once

#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <cstdlib>
#include <type_traits>
namespace wrs {

template <std::floating_point P = float, std::integral I = wrs::glsl::uint> struct Split {
    I i;
    I j;
    P spill;

    Split() : i{}, j{}, spill{} {}
    Split(I i, I j, P spill) : i(i), j(j), spill(spill) {}

    bool operator==(const Split& o) const {
        return i == o.i && j == o.j && std::abs(spill - o.spill) < 1e-8;
    }

    static constexpr glsl::StorageQualifier storage_qualifier = glsl::StorageQualifier::std430
      | glsl::StorageQualifier::std140;

    static constexpr std::size_t size(glsl::StorageQualifier storage) {
      switch (storage) {
        case glsl::StorageQualifier::std140:
        case glsl::StorageQualifier::std430:
          return sizeof(Split);
      }
    }
    static constexpr std::size_t alignment(glsl::StorageQualifier storage) {
      switch (storage) {
        case glsl::StorageQualifier::std140:
        case glsl::StorageQualifier::std430:
          return alignof(Split);
      }
    }
};
static_assert(std::regular<Split<>>);
static_assert(std::is_standard_layout_v<Split<>>);

} // namespace wrs
