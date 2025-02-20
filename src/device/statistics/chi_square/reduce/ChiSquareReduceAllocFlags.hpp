#pragma once

#include <type_traits>
namespace device {

enum class ChiSquareReduceAllocFlags {
  ALLOC_HISTOGRAM = 0x1,
  ALLOC_WEIGHTS = 0x2,
  ALLOC_CHI_SQUARE = 0x4,
  ALLOC_ALL = ALLOC_HISTOGRAM | ALLOC_WEIGHTS | ALLOC_CHI_SQUARE,
  ALLOC_NONE = 0,
};

// Enable bitwise operations for the enum
inline ChiSquareReduceAllocFlags operator|(ChiSquareReduceAllocFlags lhs, ChiSquareReduceAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareReduceAllocFlags>;
    return static_cast<ChiSquareReduceAllocFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline ChiSquareReduceAllocFlags operator&(ChiSquareReduceAllocFlags lhs, ChiSquareReduceAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareReduceAllocFlags>;
    return static_cast<ChiSquareReduceAllocFlags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline ChiSquareReduceAllocFlags operator^(ChiSquareReduceAllocFlags lhs, ChiSquareReduceAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareReduceAllocFlags>;
    return static_cast<ChiSquareReduceAllocFlags>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

inline ChiSquareReduceAllocFlags operator~(ChiSquareReduceAllocFlags flag) {
    using T = std::underlying_type_t<ChiSquareReduceAllocFlags>;
    return static_cast<ChiSquareReduceAllocFlags>(~static_cast<T>(flag));
}

// Compound assignment operators
inline ChiSquareReduceAllocFlags& operator|=(ChiSquareReduceAllocFlags& lhs, ChiSquareReduceAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline ChiSquareReduceAllocFlags& operator&=(ChiSquareReduceAllocFlags& lhs, ChiSquareReduceAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline ChiSquareReduceAllocFlags& operator^=(ChiSquareReduceAllocFlags& lhs, ChiSquareReduceAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for ChiSquareReduceAllocFlags
inline bool operator==(ChiSquareReduceAllocFlags lhs, ChiSquareReduceAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareReduceAllocFlags>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

inline bool operator!=(ChiSquareReduceAllocFlags lhs, ChiSquareReduceAllocFlags rhs) {
    return !(lhs == rhs);
}

// Comparison operators between ChiSquareReduceAllocFlags and its underlying type
inline bool operator==(ChiSquareReduceAllocFlags lhs, std::underlying_type_t<ChiSquareReduceAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<ChiSquareReduceAllocFlags>>(lhs) == rhs;
}

inline bool operator==(std::underlying_type_t<ChiSquareReduceAllocFlags> lhs, ChiSquareReduceAllocFlags rhs) {
    return rhs == lhs;
}

inline bool operator!=(ChiSquareReduceAllocFlags lhs, std::underlying_type_t<ChiSquareReduceAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(std::underlying_type_t<ChiSquareReduceAllocFlags> lhs, ChiSquareReduceAllocFlags rhs) {
    return !(lhs == rhs);
}

}
