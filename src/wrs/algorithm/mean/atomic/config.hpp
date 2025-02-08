#pragma once

#include "src/wrs/types/glsl.hpp"

namespace wrs {

class AtomicMeanConfig {
  public:
    glsl::uint workgroupSize;
    glsl::uint rows;

    constexpr AtomicMeanConfig() : workgroupSize(512), rows(8) {}
    explicit constexpr AtomicMeanConfig(glsl::uint workgroupSize, glsl::uint rows)
        : workgroupSize(workgroupSize), rows(rows) {}

    inline constexpr glsl::uint partitionSize() const {
        return workgroupSize * rows;
    }
};

struct AtomicMeanRunInfo {
    const AtomicMeanConfig config;
    const glsl::uint N;
};

}
