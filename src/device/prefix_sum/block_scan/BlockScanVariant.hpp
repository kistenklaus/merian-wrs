#pragma once

#include "src/host/types/glsl.hpp"
#include <string>

namespace device {

enum class BlockScanVariant : host::glsl::uint {
    RAKING = 1,
    RANKED = 2,
    SUBGROUP_SCAN_SHFL = 4,
    EXCLUSIVE = 8,
    INCLUSIVE = 16,
    STRIDED = 32,
    RANKED_STRIDED = RANKED | STRIDED,
    SUBGROUP_SCAN_INTRINSIC = 64,
};

// Enable bitwise operations for the enum
constexpr BlockScanVariant operator|(BlockScanVariant lhs, BlockScanVariant rhs) {
    using T = std::underlying_type_t<BlockScanVariant>;
    return static_cast<BlockScanVariant>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

constexpr BlockScanVariant operator&(BlockScanVariant lhs, BlockScanVariant rhs) {
    using T = std::underlying_type_t<BlockScanVariant>;
    return static_cast<BlockScanVariant>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

constexpr BlockScanVariant operator^(BlockScanVariant lhs, BlockScanVariant rhs) {
    using T = std::underlying_type_t<BlockScanVariant>;
    return static_cast<BlockScanVariant>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

constexpr BlockScanVariant operator~(BlockScanVariant flag) {
    using T = std::underlying_type_t<BlockScanVariant>;
    return static_cast<BlockScanVariant>(~static_cast<T>(flag));
}

// Compound assignment operators
constexpr BlockScanVariant& operator|=(BlockScanVariant& lhs, BlockScanVariant rhs) {
    return lhs = lhs | rhs;
}

constexpr BlockScanVariant& operator&=(BlockScanVariant& lhs, BlockScanVariant rhs) {
    return lhs = lhs & rhs;
}

constexpr BlockScanVariant& operator^=(BlockScanVariant& lhs, BlockScanVariant rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for BlockScanVariant
constexpr bool operator==(BlockScanVariant lhs, BlockScanVariant rhs) {
    using T = std::underlying_type_t<BlockScanVariant>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

constexpr bool operator!=(BlockScanVariant lhs, BlockScanVariant rhs) {
    return !(lhs == rhs);
}

// Comparison operators between BlockScanVariant and its underlying type
constexpr bool operator==(BlockScanVariant lhs, std::underlying_type_t<BlockScanVariant> rhs) {
    return static_cast<std::underlying_type_t<BlockScanVariant>>(lhs) == rhs;
}

constexpr bool operator==(std::underlying_type_t<BlockScanVariant> lhs, BlockScanVariant rhs) {
    return rhs == lhs;
}

constexpr bool operator!=(BlockScanVariant lhs, std::underlying_type_t<BlockScanVariant> rhs) {
    return !(lhs == rhs);
}

constexpr bool operator!=(std::underlying_type_t<BlockScanVariant> lhs, BlockScanVariant rhs) {
    return !(lhs == rhs);
}

[[maybe_unused]]
static constexpr std::string blockScanVariantName(BlockScanVariant variant) {
    if ((variant & BlockScanVariant::RAKING) == BlockScanVariant::RAKING) {
        return "RAKING";
    } else if ((variant & BlockScanVariant::RANKED) == BlockScanVariant::RANKED) {
        if ((variant & BlockScanVariant::STRIDED) == BlockScanVariant::STRIDED) {
            return "RANKED-STRIDED";
        } else {
            return "RANKED";
        }
    } else {
        return "UNNAMED";
    }
}

} // namespace device
