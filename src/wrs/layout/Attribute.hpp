#pragma once

#include "src/wrs/layout/StaticString.hpp"
namespace wrs::layout {

template <typename T, StaticString Name> struct Attribute {
    using type = T;
    static constexpr auto name = Name;
};

}
