#include "./PSAC.hpp"

#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/types/split.hpp"
#include "src/wrs/types/glsl.hpp"


using Buffers = wrs::PSAC::Buffers;
using weight_type = wrs::PSAC::weight_t;

namespace wrs::test::psac { 


wrs::pmr::AliasTable<weight_type, wrs::glsl::uint> downloadAliasTableFromStage(
    const Buffers &stage, const std::size_t weightCount, std::pmr::memory_resource *resource) {
    Buffers::AliasTableView stageView{stage.aliasTable, weightCount};

    using Entry = wrs::AliasTableEntry<weight_type, wrs::glsl::uint>;
    return stageView.download<Entry, wrs::pmr_alloc<Entry> >(resource);
}

using weight_t = float;
using Split = wrs::Split<weight_t, wrs::glsl::uint>;

std::pmr::vector<Split> downloadSplitsFromStage(
    const Buffers& stage, const std::size_t splitCount, std::pmr::memory_resource* resource) {
     Buffers::SplitsView stageView{stage.splits, splitCount}; 
     return stageView.template download<Split, wrs::pmr_alloc<Split>>(resource); 
}

}
