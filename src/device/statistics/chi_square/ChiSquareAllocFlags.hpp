#pragma once

#include <type_traits>
namespace device {

enum class ChiSquareAllocFlags {
  ALLOC_SAMPLES = 0x1,
  ALLOC_WEIGHTS = 0x2,
  ALLOC_CHI_SQUARE = 0x4,
  ALLOC_ALL = ALLOC_SAMPLES | ALLOC_WEIGHTS | ALLOC_CHI_SQUARE,
};


// Enable bitwise operations for the enum
inline ChiSquareAllocFlags operator|(ChiSquareAllocFlags lhs, ChiSquareAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareAllocFlags>;
    return static_cast<ChiSquareAllocFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline ChiSquareAllocFlags operator&(ChiSquareAllocFlags lhs, ChiSquareAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareAllocFlags>;
    return static_cast<ChiSquareAllocFlags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline ChiSquareAllocFlags operator^(ChiSquareAllocFlags lhs, ChiSquareAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareAllocFlags>;
    return static_cast<ChiSquareAllocFlags>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

inline ChiSquareAllocFlags operator~(ChiSquareAllocFlags flag) {
    using T = std::underlying_type_t<ChiSquareAllocFlags>;
    return static_cast<ChiSquareAllocFlags>(~static_cast<T>(flag));
}

// Compound assignment operators
inline ChiSquareAllocFlags& operator|=(ChiSquareAllocFlags& lhs, ChiSquareAllocFlags rhs) {
    return lhs = lhs | rhs;
}

inline ChiSquareAllocFlags& operator&=(ChiSquareAllocFlags& lhs, ChiSquareAllocFlags rhs) {
    return lhs = lhs & rhs;
}

inline ChiSquareAllocFlags& operator^=(ChiSquareAllocFlags& lhs, ChiSquareAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

// Comparison operators for ChiSquarAllocFlags
inline bool operator==(ChiSquareAllocFlags lhs, ChiSquareAllocFlags rhs) {
    using T = std::underlying_type_t<ChiSquareAllocFlags>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

inline bool operator!=(ChiSquareAllocFlags lhs, ChiSquareAllocFlags rhs) {
    return !(lhs == rhs);
}

// Comparison operators between ChiSquarAllocFlags and its underlying type
inline bool operator==(ChiSquareAllocFlags lhs, std::underlying_type_t<ChiSquareAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<ChiSquareAllocFlags>>(lhs) == rhs;
}

inline bool operator==(std::underlying_type_t<ChiSquareAllocFlags> lhs, ChiSquareAllocFlags rhs) {
    return rhs == lhs;
}

inline bool operator!=(ChiSquareAllocFlags lhs, std::underlying_type_t<ChiSquareAllocFlags> rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(std::underlying_type_t<ChiSquareAllocFlags> lhs, ChiSquareAllocFlags rhs) {
    return !(lhs == rhs);
}

}
