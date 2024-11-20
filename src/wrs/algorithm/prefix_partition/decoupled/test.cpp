#include "./test.hpp"
#include "./test/test_setup.h"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/test/test.hpp"
#include <algorithm>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

using namespace wrs::test::decoupled_prefix_partition;

template <typename weight_t>
void runTestCase(const wrs::test::TestContext& context,
                 Buffers& buffers,
                 Buffers& stage,
                 std::pmr::memory_resource* resource,
                 const TestCase& testCase) {
    // 1. Generate weights
    std::pmr::vector<weight_t> weights =
        wrs::generate_weights<weight_t, std::pmr::polymorphic_allocator<weight_t>>(
            testCase.distribution, testCase.elementCount, resource);
    weight_t pivot = testCase.getPivot<weight_t>();

    // 2. Compute refernce
    const auto [heavy, light, partition] =
        wrs::reference::pmr::partition<weight_t>(weights, pivot, resource);
    const auto heavyPrefix = wrs::reference::pmr::prefix_sum<weight_t>(heavy, resource);
    const auto lightPrefix = wrs::reference::pmr::prefix_sum<weight_t>(light, resource);
    
    
}

void wrs::test::decoupled_prefix_partition::test(const merian::ContextHandle& context) {
    TestContext c = wrs::test::setupTestContext(context);

    auto [buffers, stage, resource] = allocateBuffers(c);

    for (const auto& testCase : TEST_CASES) {
        switch (testCase.weight_type) {
        case WEIGHT_T_FLOAT:
            runTestCase<float>(c, buffers, stage, resource.get(), testCase);
            break;
        }
    }
}
