#include "./test.hpp"
#include "src/wrs/types/alias_table.hpp"
#include <vulkan/vulkan.hpp>
#include "./SampleAliasTable.hpp"

namespace wrs::test::sample_alias {

using Buffers = SampleAliasTableBuffers;

void uploadTestCase(const merian::CommandBufferHandle& cmd,
                                   const Buffers& buffers,
                                   const Buffers& stage,
                                   wrs::ImmutableAliasTableReference<float, glsl::uint> aliasTable) {
  Buffers::AliasTableView stageView{stage.aliasTable, aliasTable.size()};
  Buffers::AliasTableView localView{buffers.aliasTable, aliasTable.size()};

  stageView.upload(aliasTable);
  stageView.copyTo(cmd, localView);
  localView.expectComputeRead(cmd);
}

}
