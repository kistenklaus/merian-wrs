#pragma once

#include <type_traits>
namespace device {

enum class SplitPackAllocFlags {
    ALLOC_WEIGHTS = 0x1,
    ALLOC_PARTITION_INDICES = 0x2,
    ALLOC_PARTITION_PREFIX = 0x4,
    ALLOC_HEAVY_COUNT = 0x8,
    ALLOC_MEAN = 0x10,
    ALLOC_ALIAS_TABLE = 0x20,
    ALLOC_PARTITION_ELEMENTS = 0x40,
    ALLOC_DEFAULT = ALLOC_WEIGHTS | ALLOC_PARTITION_INDICES | ALLOC_PARTITION_PREFIX |
                    ALLOC_HEAVY_COUNT | ALLOC_MEAN | ALLOC_ALIAS_TABLE,
    ALLOC_ALL = ALLOC_DEFAULT | ALLOC_PARTITION_ELEMENTS,
    ALLOC_ONLY_INTERNALS = 0,
};

// Enable bitwise operations for SplitPackAllocFlags
inline constexpr SplitPackAllocFlags operator|(SplitPackAllocFlags lhs, SplitPackAllocFlags rhs) {
    return static_cast<SplitPackAllocFlags>(
        static_cast<std::underlying_type_t<SplitPackAllocFlags>>(lhs) |
        static_cast<std::underlying_type_t<SplitPackAllocFlags>>(rhs));
}

inline constexpr SplitPackAllocFlags operator&(SplitPackAllocFlags lhs, SplitPackAllocFlags rhs) {
    return static_cast<SplitPackAllocFlags>(
        static_cast<std::underlying_type_t<SplitPackAllocFlags>>(lhs) &
        static_cast<std::underlying_type_t<SplitPackAllocFlags>>(rhs));
}

inline constexpr SplitPackAllocFlags operator^(SplitPackAllocFlags lhs, SplitPackAllocFlags rhs) {
    return static_cast<SplitPackAllocFlags>(
        static_cast<std::underlying_type_t<SplitPackAllocFlags>>(lhs) ^
        static_cast<std::underlying_type_t<SplitPackAllocFlags>>(rhs));
}

inline constexpr SplitPackAllocFlags operator~(SplitPackAllocFlags flag) {
    return static_cast<SplitPackAllocFlags>(
        ~static_cast<std::underlying_type_t<SplitPackAllocFlags>>(flag));
}

inline constexpr SplitPackAllocFlags& operator|=(SplitPackAllocFlags& lhs,
                                                 SplitPackAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline constexpr SplitPackAllocFlags& operator&=(SplitPackAllocFlags& lhs,
                                                 SplitPackAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline constexpr SplitPackAllocFlags& operator^=(SplitPackAllocFlags& lhs,
                                                 SplitPackAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for SplitPackAllocFlags and its underlying type
inline constexpr bool operator==(SplitPackAllocFlags lhs,
                                 std::underlying_type_t<SplitPackAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<SplitPackAllocFlags>>(lhs) == rhs;
}

inline constexpr bool operator!=(SplitPackAllocFlags lhs,
                                 std::underlying_type_t<SplitPackAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline constexpr bool operator==(std::underlying_type_t<SplitPackAllocFlags> lhs,
                                 SplitPackAllocFlags rhs) {
    return rhs == lhs;
}

inline constexpr bool operator!=(std::underlying_type_t<SplitPackAllocFlags> lhs,
                                 SplitPackAllocFlags rhs) {
    return rhs != lhs;
}

} // namespace device
