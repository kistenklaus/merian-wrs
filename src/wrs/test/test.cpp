#include "./test.hpp"

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
/* #include "./compare_ranges.hpp" */
#include "./is_prefix.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_partition.hpp"
#include "src/wrs/test/is_split.hpp"
#include <fmt/base.h>
#include <fmt/format.h>
#include <functional>
#include <memory_resource>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>

wrs::test::TestContext wrs::test::setupTestContext(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
    query_pool->reset(); // LOL THIS WAS HARD TO FIND shared_ptr also defines a reset function =^).
    profiler->set_query_pool(query_pool);

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    return {
        .context = context,
        .alloc = alloc,
        .queue = queue,
        .cmdPool = cmdPool,
        .profiler = profiler,
    };
}

static void testPartitionTests(std::pmr::memory_resource* resource) {
    auto weights =
        wrs::pmr::generate_weights<float>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM, 1e4, resource);

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
        wrs::pmr::generate_weights<float>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM, 1e4, resource);

    auto prefixSum = wrs::reference::pmr::prefix_sum<float>(weights, resource);

    auto prefixError = wrs::test::pmr::assert_is_inclusive_prefix<float>(weights, prefixSum);
    if (prefixError) {
        SPDLOG_ERROR(prefixError.message());
        throw std::runtime_error("Test of test failed: wrs::reference::prefix_sum or "
                                 "wrs::test::assert_is_inclusive_prefix is wrong");
    }
}

static void testReduceReference(std::pmr::memory_resource* resource) {
    {
        constexpr size_t ELEM_COUNT = 1e4;
        auto elements =
            wrs::pmr::generate_weights<float>(wrs::Distribution::UNIFORM, ELEM_COUNT, resource);

        auto reduction = wrs::reference::pmr::reduce<float>(elements, resource);
        if (std::abs(reduction - static_cast<float>(ELEM_COUNT)) > 0.01) {
            throw std::runtime_error("Test of test failed: wrs::reference::reduce is wrong!");
        }
    }

    {
        constexpr size_t ELEM_COUNT = 1e4;
        auto elements = wrs::pmr::generate_weights<float>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
                                                          ELEM_COUNT, resource);

        auto reduction = wrs::reference::pmr::reduce<float>(elements, resource);
        auto stdReduction =
            std::accumulate(elements.begin(), elements.end(), 0.0f, std::plus<float>());
        if (std::abs(reduction - stdReduction) > 0.01) {
            throw std::runtime_error("Test of test failed: wrs::reference::reduce is wrong!");
        }
    }
}

static void testSplitTests(std::pmr::memory_resource* resource) {

    size_t N = 1024 * 2048;
    size_t K = 2048;
    std::pmr::vector<float> weights =
        wrs::pmr::generate_weights<float>(wrs::Distribution::SEEDED_RANDOM_UNIFORM, N, resource);

    float reduction = wrs::reference::reduce<float>(weights);
    float average = reduction / static_cast<float>(weights.size());

    auto [heavy, light, storage] =
        wrs::reference::pmr::stable_partition<float>(weights, average, resource);

    std::pmr::vector<float> heavyPrefix =
        wrs::reference::pmr::prefix_sum<float>(heavy, false, resource);
    std::pmr::vector<float> lightPrefix =
        wrs::reference::pmr::prefix_sum<float>(light, false, resource);

    std::pmr::vector<wrs::test::internal::Split<float>> splits =
        wrs::reference::pmr::splitK<float>(heavyPrefix, lightPrefix, average, N, K, resource);

    auto err = wrs::test::pmr::assert_is_split<float>(splits, K, heavyPrefix, lightPrefix, average,
                                                      0.01, resource);
    if (err) {
        SPDLOG_ERROR(fmt::format("Test of tests failed: split of it's assertion are invalid\n{}",
                                 err.message()));
    }

    /* wrs::reference::splitK(const std::span<T> heavyPrefix, const std::span<T> lightPrefix, T
     * mean, size_t N, size_t K) */
}

void wrs::test::testTests() {
    SPDLOG_DEBUG("Testing tests...");
    wrs::memory::StackResource stackResource{10000 * sizeof(float)};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource resource{&fallbackResource};

    SPDLOG_DEBUG("Testing reduce reference");
    stackResource.reset();
    testReduceReference(&resource);

    SPDLOG_DEBUG("Testing partition assertions tests");
    stackResource.reset();
    testPartitionTests(&resource);

    SPDLOG_DEBUG("Testing inclusive prefix assertion tests");
    stackResource.reset();
    testPrefixTests(&resource);

    SPDLOG_DEBUG("Testing split assertion tests");
    stackResource.reset();
    testSplitTests(&resource);

    SPDLOG_INFO("Tested tests and references successfully!");
}
