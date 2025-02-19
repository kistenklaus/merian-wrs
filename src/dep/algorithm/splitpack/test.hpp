#pragma once

#include "merian/vk/context.hpp"
#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/types/glsl.hpp"
#include "src/wrs/types/split.hpp"
namespace wrs::test::splitpack {

struct Results {
    wrs::pmr::AliasTable<float, glsl::uint> aliasTable;
    std::pmr::vector<wrs::Split<float, glsl::uint>> splits;
};

void test(const merian::ContextHandle& context);

} // namespace wrs::test::splitpack
