#include "./test.hpp"

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
/* #include "./compare_ranges.hpp" */
#include "./is_prefix.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/test/is_partition.hpp"
#include <memory_resource>
#include <spdlog/spdlog.h>

wrs::test::TestContext wrs::test::setupTestContext(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
    query_pool.reset();
    profiler->set_query_pool(query_pool);

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {1, 3, 6};
    assert_is_inclusive_prefix<float>(std::span(a), std::span(b));
    /* compare_ranges(a, b); */

    return {
        .alloc = alloc,
        .queue = queue,
        .cmdPool = cmdPool,
        .profiler = profiler,
        .__profilerQueryPool = query_pool,
    };
}

static void testPartitionTests(std::pmr::memory_resource* resource) {
    auto weights =
        wrs::pmr::generate_weights<float>(wrs::Distribution::SEEDED_RANDOM_UNIFORM, 256, resource);

    float pivot = weights.back();

    const auto [heavy, light, partitionStorage] =
        wrs::reference::pmr::partition<float>(weights, pivot, resource);

    auto partitionError =
        wrs::test::pmr::assert_is_partition<float>(heavy, light, weights, pivot, resource);
    if (partitionError) {
        throw std::runtime_error("Test of test failed: wrs::reference::pmr::partition or "
                                 "wrs::test::pmr::assert_is_partition is wrong");
    }

    std::pmr::vector<float> invalidHeavy{heavy.begin(), heavy.end(), resource};
    std::pmr::vector<float> invalidLight{light.begin(), light.end(), resource};
    float bh = invalidHeavy.back();
    invalidHeavy.pop_back();
    float bl = invalidLight.back();
    invalidLight.pop_back();
    invalidHeavy.push_back(bl);
    invalidLight.push_back(bh);

    auto assertError = wrs::test::pmr::assert_is_partition<float>(invalidHeavy, invalidLight,
                                                                  weights, pivot, resource);
    if (!assertError) {
        throw std::runtime_error("Test of test failed: wrs::reference::pmr::partition or "
                                 "wrs::test::pmr::assert_is_partition is wrong");
    }
}

static void testPrefixTests(std::pmr::memory_resource* resource) {
    auto weights =
        wrs::pmr::generate_weights<float>(wrs::Distribution::SEEDED_RANDOM_UNIFORM, 1e7, resource);

    auto prefixSum = wrs::reference::pmr::prefix_sum<float>(weights, resource);

    auto prefixError = wrs::test::pmr::assert_is_inclusive_prefix<float>(weights, prefixSum);
    if (prefixError) {
        SPDLOG_ERROR(prefixError.message());
        throw std::runtime_error("Test of test failed: wrs::reference::prefix_sum or "
                                 "wrs::test::assert_is_inclusive_prefix is wrong");
    }
}

void wrs::test::testTests() {
    std::pmr::monotonic_buffer_resource resource{4096};
    SPDLOG_INFO("Testing tests...");
    testPartitionTests(&resource);
    SPDLOG_INFO("Partition tests tested successfully");
    testPrefixTests(&resource);
    SPDLOG_INFO("Inclusive Prefix sum tests tested successfully");
}
