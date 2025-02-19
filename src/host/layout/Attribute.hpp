#pragma once

#include "src/host/layout/StaticString.hpp"

namespace host::layout {

template <typename T, StaticString Name> struct Attribute {
    using type = T;
    static constexpr auto name = Name;
};

}
