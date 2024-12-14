#pragma once

#include "src/wrs/algorithm/pack/scalar/test/test_types.hpp"
#include "src/wrs/test/test.hpp"
#include <tuple>
namespace wrs::test::scalar_pack {


inline std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
  Buffers buffers;
  Buffers stage;

  return std::make_tuple(buffers, stage);
}

}
