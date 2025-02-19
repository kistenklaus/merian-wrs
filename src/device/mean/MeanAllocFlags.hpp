#pragma once

#include <type_traits>
namespace device {

enum class MeanAllocFlags {
    ALLOC_ELEMENTS = 1,
    ALLOC_MEAN = 2,
    ALLOC_ALL = ALLOC_ELEMENTS | ALLOC_MEAN,
};

// Thank you GPT .....
constexpr MeanAllocFlags operator|(MeanAllocFlags lhs, MeanAllocFlags rhs) {
    using T = std::underlying_type_t<MeanAllocFlags>;
    return static_cast<MeanAllocFlags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

constexpr MeanAllocFlags operator&(MeanAllocFlags lhs, MeanAllocFlags rhs) {
    using T = std::underlying_type_t<MeanAllocFlags>;
    return static_cast<MeanAllocFlags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

constexpr MeanAllocFlags operator^(MeanAllocFlags lhs, MeanAllocFlags rhs) {
    using T = std::underlying_type_t<MeanAllocFlags>;
    return static_cast<MeanAllocFlags>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
}

constexpr MeanAllocFlags operator~(MeanAllocFlags flag) {
    using T = std::underlying_type_t<MeanAllocFlags>;
    return static_cast<MeanAllocFlags>(~static_cast<T>(flag));
}

constexpr MeanAllocFlags& operator|=(MeanAllocFlags& lhs, MeanAllocFlags rhs) {
    return lhs = lhs | rhs;
}

constexpr MeanAllocFlags& operator&=(MeanAllocFlags& lhs, MeanAllocFlags rhs) {
    return lhs = lhs & rhs;
}

constexpr MeanAllocFlags& operator^=(MeanAllocFlags& lhs, MeanAllocFlags rhs) {
    return lhs = lhs ^ rhs;
}

constexpr bool operator==(MeanAllocFlags lhs, MeanAllocFlags rhs) {
    using T = std::underlying_type_t<MeanAllocFlags>;
    return static_cast<T>(lhs) == static_cast<T>(rhs);
}

constexpr bool operator!=(MeanAllocFlags lhs, MeanAllocFlags rhs) {
    return !(lhs == rhs);
}

// Comparison operators between MeanAllocFlags and its underlying type
constexpr bool operator==(MeanAllocFlags lhs, std::underlying_type_t<MeanAllocFlags> rhs) {
    return static_cast<std::underlying_type_t<MeanAllocFlags>>(lhs) == rhs;
}

constexpr bool operator==(std::underlying_type_t<MeanAllocFlags> lhs, MeanAllocFlags rhs) {
    return rhs == lhs;
}

constexpr bool operator!=(MeanAllocFlags lhs, std::underlying_type_t<MeanAllocFlags> rhs) {
    return !(lhs == rhs);
}

constexpr bool operator!=(std::underlying_type_t<MeanAllocFlags> lhs, MeanAllocFlags rhs) {
    return !(lhs == rhs);
}

} // namespace wrs
