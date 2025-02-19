#pragma once

#include <type_traits>
namespace device {

enum class PartitionAllocFlags {
    ALLOC_ELEMENTS = 0x1,
    ALLOC_PIVOT = 0x2,
    ALLOC_PARTITION_INDICES = 0x4,
    ALLOC_PARTITION_ELEMENTS = 0x8,
    ALLOC_HEAVY_COUNT = 0x10,
    ALLOC_ALL =
        ALLOC_ELEMENTS | ALLOC_PIVOT | ALLOC_PARTITION_INDICES | ALLOC_PARTITION_ELEMENTS | ALLOC_HEAVY_COUNT,
};

// Enable bitwise operations for the enum
inline PartitionAllocFlags operator|(PartitionAllocFlags lhs, PartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PartitionAllocFlags>;
    return static_cast<PartitionAllocFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline PartitionAllocFlags operator&(PartitionAllocFlags lhs, PartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PartitionAllocFlags>;
    return static_cast<PartitionAllocFlags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline PartitionAllocFlags operator^(PartitionAllocFlags lhs, PartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PartitionAllocFlags>;
    return static_cast<PartitionAllocFlags>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

inline PartitionAllocFlags operator~(PartitionAllocFlags flag) {
    using T = std::underlying_type_t<PartitionAllocFlags>;
    return static_cast<PartitionAllocFlags>(~static_cast<T>(flag));
}

// Compound assignment operators
inline PartitionAllocFlags& operator|=(PartitionAllocFlags& lhs, PartitionAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline PartitionAllocFlags& operator&=(PartitionAllocFlags& lhs, PartitionAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline PartitionAllocFlags& operator^=(PartitionAllocFlags& lhs, PartitionAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for PartitionAllocFlags
inline bool operator==(PartitionAllocFlags lhs, PartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PartitionAllocFlags>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

inline bool operator!=(PartitionAllocFlags lhs, PartitionAllocFlags rhs) {
    return !(lhs == rhs);
}

// Comparison operators between PartitionAllocFlags and its underlying type
inline bool operator==(PartitionAllocFlags lhs, std::underlying_type_t<PartitionAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<PartitionAllocFlags>>(lhs) == rhs;
}

inline bool operator==(std::underlying_type_t<PartitionAllocFlags> lhs, PartitionAllocFlags rhs) {
    return rhs == lhs;
}

inline bool operator!=(PartitionAllocFlags lhs, std::underlying_type_t<PartitionAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(std::underlying_type_t<PartitionAllocFlags> lhs, PartitionAllocFlags rhs) {
    return !(lhs == rhs);
}

} // namespace device
