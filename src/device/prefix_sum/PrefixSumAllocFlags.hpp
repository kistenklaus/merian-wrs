#pragma once

#include <type_traits>
namespace device {

enum class PrefixSumAllocFlags {
    ALLOC_ELEMENTS = 0x1,
    ALLOC_PREFIX_SUM = 0x2,
    ALLOC_ALL = ALLOC_ELEMENTS | ALLOC_PREFIX_SUM,
};

// Enable bitwise operations for the enum
inline PrefixSumAllocFlags operator|(PrefixSumAllocFlags lhs, PrefixSumAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixSumAllocFlags>;
    return static_cast<PrefixSumAllocFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline PrefixSumAllocFlags operator&(PrefixSumAllocFlags lhs, PrefixSumAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixSumAllocFlags>;
    return static_cast<PrefixSumAllocFlags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline PrefixSumAllocFlags operator^(PrefixSumAllocFlags lhs, PrefixSumAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixSumAllocFlags>;
    return static_cast<PrefixSumAllocFlags>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

inline PrefixSumAllocFlags operator~(PrefixSumAllocFlags flag) {
    using T = std::underlying_type_t<PrefixSumAllocFlags>;
    return static_cast<PrefixSumAllocFlags>(~static_cast<T>(flag));
}

// Compound assignment operators
inline PrefixSumAllocFlags& operator|=(PrefixSumAllocFlags& lhs, PrefixSumAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline PrefixSumAllocFlags& operator&=(PrefixSumAllocFlags& lhs, PrefixSumAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline PrefixSumAllocFlags& operator^=(PrefixSumAllocFlags& lhs, PrefixSumAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for PrefixSumAllocFlags
inline bool operator==(PrefixSumAllocFlags lhs, PrefixSumAllocFlags rhs) {
    using T = std::underlying_type_t<PrefixSumAllocFlags>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

inline bool operator!=(PrefixSumAllocFlags lhs, PrefixSumAllocFlags rhs) {
    return !(lhs == rhs);
}

// Comparison operators between PrefixSumAllocFlags and its underlying type
inline bool operator==(PrefixSumAllocFlags lhs, std::underlying_type_t<PrefixSumAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<PrefixSumAllocFlags>>(lhs) == rhs;
}

inline bool operator==(std::underlying_type_t<PrefixSumAllocFlags> lhs, PrefixSumAllocFlags rhs) {
    return rhs == lhs;
}

inline bool operator!=(PrefixSumAllocFlags lhs, std::underlying_type_t<PrefixSumAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(std::underlying_type_t<PrefixSumAllocFlags> lhs, PrefixSumAllocFlags rhs) {
    return !(lhs == rhs);
}

} // namespace device
