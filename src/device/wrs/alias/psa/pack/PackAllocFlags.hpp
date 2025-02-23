#pragma once

#include <type_traits>
namespace device {

enum class PackAllocFlags {
    ALLOC_WEIGHTS = 0x1,
    ALLOC_MEAN = 0x2,
    ALLOC_HEAVY_COUNT = 0x4,
    ALLOC_PARTITION_INDICES = 0x8,
    ALLOC_SPLITS = 0x10,
    ALLOC_ALIAS_TABLE = 0x20,
    ALLOC_PARTITION_ELEMENTS = 0x40,
    ALLOC_DEFAULT = ALLOC_WEIGHTS | ALLOC_MEAN | ALLOC_HEAVY_COUNT | ALLOC_PARTITION_INDICES |
                    ALLOC_SPLITS | ALLOC_ALIAS_TABLE,
    ALLOC_ALL = ALLOC_DEFAULT | ALLOC_PARTITION_ELEMENTS,
    ALLOC_NONE = 0x0,
};

// Enable bitwise operations for PackAllocFlags
inline constexpr PackAllocFlags operator|(PackAllocFlags lhs, PackAllocFlags rhs) {
    return static_cast<PackAllocFlags>(static_cast<std::underlying_type_t<PackAllocFlags>>(lhs) |
                                       static_cast<std::underlying_type_t<PackAllocFlags>>(rhs));
}

inline constexpr PackAllocFlags operator&(PackAllocFlags lhs, PackAllocFlags rhs) {
    return static_cast<PackAllocFlags>(static_cast<std::underlying_type_t<PackAllocFlags>>(lhs) &
                                       static_cast<std::underlying_type_t<PackAllocFlags>>(rhs));
}

inline constexpr PackAllocFlags operator^(PackAllocFlags lhs, PackAllocFlags rhs) {
    return static_cast<PackAllocFlags>(static_cast<std::underlying_type_t<PackAllocFlags>>(lhs) ^
                                       static_cast<std::underlying_type_t<PackAllocFlags>>(rhs));
}

inline constexpr PackAllocFlags operator~(PackAllocFlags flag) {
    return static_cast<PackAllocFlags>(~static_cast<std::underlying_type_t<PackAllocFlags>>(flag));
}

inline constexpr PackAllocFlags& operator|=(PackAllocFlags& lhs, PackAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline constexpr PackAllocFlags& operator&=(PackAllocFlags& lhs, PackAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline constexpr PackAllocFlags& operator^=(PackAllocFlags& lhs, PackAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for PackAllocFlags and its underlying type
inline constexpr bool operator==(PackAllocFlags lhs, std::underlying_type_t<PackAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<PackAllocFlags>>(lhs) == rhs;
}

inline constexpr bool operator!=(PackAllocFlags lhs, std::underlying_type_t<PackAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline constexpr bool operator==(std::underlying_type_t<PackAllocFlags> lhs, PackAllocFlags rhs) {
    return rhs == lhs;
}

inline constexpr bool operator!=(std::underlying_type_t<PackAllocFlags> lhs, PackAllocFlags rhs) {
    return rhs != lhs;
}

} // namespace device
