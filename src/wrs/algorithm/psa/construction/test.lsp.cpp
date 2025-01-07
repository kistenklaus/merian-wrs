#include "./PSAC.hpp"

#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/layout/BufferView.hpp"


using Buffers = wrs::PSAC::Buffers;
using weight_type = wrs::PSAC::weight_t;

namespace wrs::test::psac { 


wrs::pmr::AliasTable<weight_type, wrs::glsl::uint> downloadAliasTableFromStage(
    const Buffers &stage, const std::size_t weightCount, std::pmr::memory_resource *resource) {
    Buffers::AliasTableView stageView{stage.aliasTable, weightCount};

    using Entry = wrs::AliasTableEntry<weight_type, wrs::glsl::uint>;
    return stageView.download<Entry, wrs::pmr_alloc<Entry> >(resource);
}

}
