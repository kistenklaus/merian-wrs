#pragma once
#include <algorithm>
#include <string_view>
namespace wrs::layout {


template <std::size_t N> struct StaticString {
    char value[N];

    // ReSharper disable once CppNonExplicitConvertingConstructor
    consteval StaticString(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }

    consteval operator std::string_view() const {
        return std::string_view(value, N - 1); // Exclude null terminator
    }
};

}
