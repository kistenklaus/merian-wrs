#pragma once

#include <type_traits>
namespace device {

enum class SplitAllocFlags {
    ALLOC_PARTITION_PREFIX = 0x1,
    ALLOC_MEAN = 0x2,
    ALLOC_SPLITS = 0x4,
    ALLOC_HEAVY_COUNT = 0x8,
    ALLOC_ALL = ALLOC_PARTITION_PREFIX | ALLOC_MEAN | ALLOC_SPLITS | ALLOC_HEAVY_COUNT,
    ALLOC_NONE = 0x0,
};

// Enable bitwise operations for SplitAllocFlags
inline constexpr SplitAllocFlags operator|(SplitAllocFlags lhs, SplitAllocFlags rhs) {
    return static_cast<SplitAllocFlags>(static_cast<std::underlying_type_t<SplitAllocFlags>>(lhs) |
                                        static_cast<std::underlying_type_t<SplitAllocFlags>>(rhs));
}

inline constexpr SplitAllocFlags operator&(SplitAllocFlags lhs, SplitAllocFlags rhs) {
    return static_cast<SplitAllocFlags>(static_cast<std::underlying_type_t<SplitAllocFlags>>(lhs) &
                                        static_cast<std::underlying_type_t<SplitAllocFlags>>(rhs));
}

inline constexpr SplitAllocFlags operator^(SplitAllocFlags lhs, SplitAllocFlags rhs) {
    return static_cast<SplitAllocFlags>(static_cast<std::underlying_type_t<SplitAllocFlags>>(lhs) ^
                                        static_cast<std::underlying_type_t<SplitAllocFlags>>(rhs));
}

inline constexpr SplitAllocFlags operator~(SplitAllocFlags flag) {
    return static_cast<SplitAllocFlags>(
        ~static_cast<std::underlying_type_t<SplitAllocFlags>>(flag));
}

inline constexpr SplitAllocFlags& operator|=(SplitAllocFlags& lhs, SplitAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline constexpr SplitAllocFlags& operator&=(SplitAllocFlags& lhs, SplitAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline constexpr SplitAllocFlags& operator^=(SplitAllocFlags& lhs, SplitAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for SplitAllocFlags and its underlying type
inline constexpr bool operator==(SplitAllocFlags lhs, std::underlying_type_t<SplitAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<SplitAllocFlags>>(lhs) == rhs;
}

inline constexpr bool operator!=(SplitAllocFlags lhs, std::underlying_type_t<SplitAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline constexpr bool operator==(std::underlying_type_t<SplitAllocFlags> lhs, SplitAllocFlags rhs) {
    return rhs == lhs;
}

inline constexpr bool operator!=(std::underlying_type_t<SplitAllocFlags> lhs, SplitAllocFlags rhs) {
    return rhs != lhs;
}

} // namespace device
