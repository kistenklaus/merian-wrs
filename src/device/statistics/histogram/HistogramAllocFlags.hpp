#pragma once

#include <type_traits>
namespace device {

enum class HistogramAllocFlags {
    ALLOC_SAMPLES = 0x1,
    ALLOC_HISTOGRAM = 0x2,
    ALLOC_ALL = ALLOC_SAMPLES | ALLOC_HISTOGRAM,
    ALLOC_NONE = 0x0,
};

// Enable bitwise operations for the enum
inline HistogramAllocFlags operator|(HistogramAllocFlags lhs, HistogramAllocFlags rhs) {
    using T = std::underlying_type_t<HistogramAllocFlags>;
    return static_cast<HistogramAllocFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline HistogramAllocFlags operator&(HistogramAllocFlags lhs, HistogramAllocFlags rhs) {
    using T = std::underlying_type_t<HistogramAllocFlags>;
    return static_cast<HistogramAllocFlags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline HistogramAllocFlags operator^(HistogramAllocFlags lhs, HistogramAllocFlags rhs) {
    using T = std::underlying_type_t<HistogramAllocFlags>;
    return static_cast<HistogramAllocFlags>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

inline HistogramAllocFlags operator~(HistogramAllocFlags flag) {
    using T = std::underlying_type_t<HistogramAllocFlags>;
    return static_cast<HistogramAllocFlags>(~static_cast<T>(flag));
}

// Compound assignment operators
inline HistogramAllocFlags& operator|=(HistogramAllocFlags& lhs, HistogramAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline HistogramAllocFlags& operator&=(HistogramAllocFlags& lhs, HistogramAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline HistogramAllocFlags& operator^=(HistogramAllocFlags& lhs, HistogramAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for HistogramAllocFlags
inline bool operator==(HistogramAllocFlags lhs, HistogramAllocFlags rhs) {
    using T = std::underlying_type_t<HistogramAllocFlags>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

inline bool operator!=(HistogramAllocFlags lhs, HistogramAllocFlags rhs) {
    return !(lhs == rhs);
}

// Comparison operators between HistogramAllocFlags and its underlying type
inline bool operator==(HistogramAllocFlags lhs, std::underlying_type_t<HistogramAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<HistogramAllocFlags>>(lhs) == rhs;
}

inline bool operator==(std::underlying_type_t<HistogramAllocFlags> lhs, HistogramAllocFlags rhs) {
    return rhs == lhs;
}

inline bool operator!=(HistogramAllocFlags lhs, std::underlying_type_t<HistogramAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(std::underlying_type_t<HistogramAllocFlags> lhs, HistogramAllocFlags rhs) {
    return !(lhs == rhs);
}

}; // namespace device
