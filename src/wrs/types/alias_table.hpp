#pragma once

#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/types/glsl.hpp"
#include "src/wrs/why.hpp"
#include <concepts>
#include <span>
#include <type_traits>
#include <vector>
namespace wrs {

template <std::floating_point P = float, std::integral I = wrs::glsl::uint> struct AliasTableEntry {
    P p;
    I a;

    AliasTableEntry() = default;

    AliasTableEntry(P p, I a) : p(p), a(a) {}

    bool operator==(const AliasTableEntry& o) const {
        return o == o.a && std::abs(p - o.p) < 1e-8;
    }
};
static_assert(std::regular<AliasTableEntry<float, wrs::glsl::uint>>);
static_assert(std::is_trivial_v<AliasTableEntry<>>);
static_assert(std::is_standard_layout_v<AliasTableEntry<>>);

template <std::floating_point P = float,
          std::integral I = wrs::glsl::uint,
          wrs::typed_allocator<AliasTableEntry<P, I>> Allocator =
              std::allocator<AliasTableEntry<P, I>>>
using AliasTable = std::vector<AliasTableEntry<P, I>, Allocator>;

template <std::floating_point P = float,
          std::integral I = wrs::glsl::uint>
using ImmutableAliasTableReference = std::span<const AliasTableEntry<P,I>>;

namespace pmr {

template <std::floating_point P = float, std::integral I = wrs::glsl::uint>
using AliasTable = std::pmr::vector<AliasTableEntry<P, I>>;
}

} // namespace wrs
