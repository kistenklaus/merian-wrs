#include <vulkan/vulkan.hpp>
#include <span>
#include "src/wrs/types/split.hpp"
#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/algorithm/pack/scalar/ScalarPack.hpp"

namespace wrs::test::scalar_pack {

using weight_t = float;
using Buffers = ScalarPackBuffers;

void uploadSplits(const vk::CommandBuffer cmd,
                         std::span<const wrs::Split<weight_t, wrs::glsl::uint>> splits,
                         Buffers& buffers,
                         Buffers& stage) {
    Buffers::SplitsView stageView{stage.splits, splits.size()};
    Buffers::SplitsView localView{buffers.splits, splits.size()};
    stageView.upload<wrs::Split<weight_t, wrs::glsl::uint>>(splits);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

wrs::pmr::AliasTable<weight_t, wrs::glsl::uint>
downloadAliasTableFromStage(std::size_t N, Buffers& stage, std::pmr::memory_resource* resource) {
    Buffers::AliasTableView stageView{stage.aliasTable, N};
    using Entry = wrs::AliasTableEntry<weight_t, wrs::glsl::uint>;
    return stageView.download<Entry, wrs::pmr_alloc<Entry>>(resource);
}

}
