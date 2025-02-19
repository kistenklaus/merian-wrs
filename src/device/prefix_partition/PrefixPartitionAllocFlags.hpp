#pragma once

#include <type_traits>
namespace device {

enum class PrefixPartitionAllocFlags {
    ALLOC_ELEMENTS = 0x1,
    ALLOC_PIVOT = 0x2,
    ALLOC_PARTITION_INDICES = 0x4,
    ALLOC_PARTITION_ELEMENTS = 0x8,
    ALLOC_PARTITION_PREFIX = 0x10,
    ALLOC_HEAVY_COUNT = 0x20,
    ALLOC_ALL = ALLOC_ELEMENTS | ALLOC_PIVOT | ALLOC_PARTITION_INDICES | ALLOC_PARTITION_ELEMENTS |
                ALLOC_PARTITION_PREFIX | ALLOC_HEAVY_COUNT,
};

// Enable bitwise operations for the enum
inline PrefixPartitionAllocFlags operator|(PrefixPartitionAllocFlags lhs,
                                           PrefixPartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixPartitionAllocFlags>;
    return static_cast<PrefixPartitionAllocFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline PrefixPartitionAllocFlags operator&(PrefixPartitionAllocFlags lhs,
                                           PrefixPartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixPartitionAllocFlags>;
    return static_cast<PrefixPartitionAllocFlags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline PrefixPartitionAllocFlags operator^(PrefixPartitionAllocFlags lhs,
                                           PrefixPartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixPartitionAllocFlags>;
    return static_cast<PrefixPartitionAllocFlags>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

inline PrefixPartitionAllocFlags operator~(PrefixPartitionAllocFlags flag) {
    using T = std::underlying_type_t<PrefixPartitionAllocFlags>;
    return static_cast<PrefixPartitionAllocFlags>(~static_cast<T>(flag));
}

// Compound assignment operators
inline PrefixPartitionAllocFlags& operator|=(PrefixPartitionAllocFlags& lhs,
                                             PrefixPartitionAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline PrefixPartitionAllocFlags& operator&=(PrefixPartitionAllocFlags& lhs,
                                             PrefixPartitionAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline PrefixPartitionAllocFlags& operator^=(PrefixPartitionAllocFlags& lhs,
                                             PrefixPartitionAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for PrefixPartitionAllocFlags
inline bool operator==(PrefixPartitionAllocFlags lhs, PrefixPartitionAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixPartitionAllocFlags>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

inline bool operator!=(PrefixPartitionAllocFlags lhs, PrefixPartitionAllocFlags rhs) {
    return !(lhs == rhs);
}

// Comparison operators between PrefixPartitionAllocFlags and its underlying type
inline bool operator==(PrefixPartitionAllocFlags lhs,
                       std::underlying_type_t<PrefixPartitionAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<PrefixPartitionAllocFlags>>(lhs) == rhs;
}

inline bool operator==(std::underlying_type_t<PrefixPartitionAllocFlags> lhs,
                       PrefixPartitionAllocFlags rhs) {
    return rhs == lhs;
}

inline bool operator!=(PrefixPartitionAllocFlags lhs,
                       std::underlying_type_t<PrefixPartitionAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(std::underlying_type_t<PrefixPartitionAllocFlags> lhs,
                       PrefixPartitionAllocFlags rhs) {
    return !(lhs == rhs);
}

} // namespace device
