#pragma once

/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : HSTRepr.hpp
 *
 * Array representation of the reduction tree.
 * Primarily used by the hiearichcal sampling approach, for other methods,
 * there are definitely more compact methods to represent this!
 */

#include "src/wrs/types/glsl.hpp"
#include <array>
#include <cstddef>
#include <fmt/base.h>
#include <spdlog/spdlog.h>
namespace wrs::hst {

class HSTRepr {
  public:
    struct HSTLevel {
        glsl::uint childOffset;
        glsl::uint numChildren;
        glsl::uint parentOffset;
        glsl::uint numParents;
        bool overlap;
    };

    HSTRepr(std::size_t N) {
        glsl::uint childOffset = 0;
        glsl::uint numChildren = N;
        glsl::uint parentOffset = N & (~0x1); // round down to multiple of 2
        glsl::uint numParents = (N + 2 - 1) / 2;
        bool overlap = N & 0x1;

        glsl::uint level = 0;
        while (numParents > 1) {
            m_levels[level++] = HSTLevel{
                .childOffset = childOffset,
                .numChildren = numChildren,
                .parentOffset = parentOffset,
                .numParents = numParents,
                .overlap = overlap,
            };

            childOffset = parentOffset;
            numChildren = numParents;
            parentOffset += numParents & (~0x1);
            overlap = numParents & 0x1;
            numParents = (numParents + 2 - 1) / 2;
        }
        m_numLevels = level;
    }

    std::span<const HSTLevel> get() const {
        return std::span(m_levels.begin(), m_numLevels);
    }

    std::size_t size() {
        return get().back().parentOffset + get().back().numParents;
    }

  private:
    std::array<HSTLevel, 30> m_levels;
    std::size_t m_numLevels;
};

} // namespace wrs::hst
