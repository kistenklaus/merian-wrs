#include "./test.hpp"

#include "./is_prefix.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler_shaderc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/eval/chi_square.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/psa_alias_table.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/reference/sweeping_alias_table.hpp"
#include "src/wrs/test/is_alias_table.hpp"
#include "src/wrs/test/is_partition.hpp"
#include "src/wrs/test/is_split.hpp"
#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/types/split.hpp"
#include <cassert>
#include <fmt/base.h>
#include <fmt/format.h>
#include <functional>
#include <memory_resource>
#include <numeric>
#include <random>
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

    merian::ShaderCompilerHandle shaderCompiler = std::make_shared<merian::SystemGlslcCompiler>(context);

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    return {
        .context = context,
        .alloc = alloc,
        .queue = queue,
        .cmdPool = cmdPool,
        .profiler = profiler,
        .shaderCompiler = shaderCompiler,
    };
}

static void testPartitionTests(std::pmr::memory_resource* resource) {
    SPDLOG_DEBUG("Testing wrs::reference::pmr::partition");
    auto weights =
        wrs::pmr::generate_weights<float>(wrs::Distribution::SEEDED_RANDOM_UNIFORM, 10, resource);

    auto average = wrs::reference::pmr::tree_reduction<float>(weights, resource) /
                   static_cast<float>(weights.size());
    float pivot = average;

    const auto heavyLightPartition =
        wrs::reference::pmr::partition<float>(weights, pivot, resource);

    auto partitionError = wrs::test::pmr::assert_is_partition<float>(
        heavyLightPartition.heavy(), heavyLightPartition.light(), weights, pivot, resource);
    if (partitionError) {
        SPDLOG_ERROR(partitionError.message());
        throw std::runtime_error("Test of test failed: wrs::reference::pmr::partition or "
                                 "wrs::test::pmr::assert_is_partition is wrong. Expected no error");
    }

    std::pmr::vector<float> invalidHeavy{heavyLightPartition.heavy().begin(),
                                         heavyLightPartition.heavy().end(), resource};
    std::pmr::vector<float> invalidLight{heavyLightPartition.light().begin(),
                                         heavyLightPartition.light().end(), resource};
    float bh = invalidHeavy.back();
    invalidHeavy.pop_back();
    float bl = invalidLight.back();
    invalidLight.pop_back();
    invalidHeavy.push_back(bl);
    invalidLight.push_back(bh);

    auto assertError = wrs::test::pmr::assert_is_partition<float>(invalidHeavy, invalidLight,
                                                                  weights, pivot, resource);
    if (!assertError) {

        SPDLOG_ERROR(assertError.message());
        throw std::runtime_error("Test of test failed: wrs::reference::pmr::partition or "
                                 "wrs::test::pmr::assert_is_partition is wrong. Expected error");
    }

    { // Test stable_partition_indices

        SPDLOG_DEBUG("Testing wrs::reference::pmr::stable_partition_indices");
        const auto heavyLightIndicies =
            wrs::reference::pmr::stable_partition_indicies<float, uint32_t>(weights, pivot,
                                                                            resource);
        const auto heavyIndices = heavyLightIndicies.heavy();
        const auto lightIndices = heavyLightIndicies.light();
        std::pmr::vector<float> heavyPartition{heavyIndices.size(), resource};
        std::pmr::vector<float> lightPartition{lightIndices.size(), resource};
        for (size_t i = 0; i < heavyIndices.size(); ++i) {
            heavyPartition[i] = weights[heavyIndices[i]];
        }
        for (size_t i = 0; i < lightIndices.size(); ++i) {
            lightPartition[i] = weights[lightIndices[i]];
        }
        auto err = wrs::test::pmr::assert_is_partition<float>(heavyPartition, lightPartition,
                                                              weights, pivot);
        if (err) {
            SPDLOG_ERROR(err.message());
            throw std::runtime_error(
                "Test of test failed: wrs::reference::pmr::stable_partition_indices is wrong!");
        }

        SPDLOG_DEBUG("Testing wrs::reference::pmr::stable_partition");

        const auto heavyLightPartition =
            wrs::reference::pmr::stable_partition<float>(weights, pivot, resource);
        const auto rh = heavyLightPartition.heavy();
        const auto rl = heavyLightPartition.light();
        assert(rh.size() == heavyIndices.size());
        for (size_t i = 0; i < heavyPartition.size(); ++i) {
            assert(rh[i] == heavyPartition[i]);
        }
        for (size_t i = 0; i < lightPartition.size(); ++i) {
            assert(rl[i] == lightPartition[i]);
        }
    }
}

static void testPrefixTests(std::pmr::memory_resource* resource) {
    using T = float;
    constexpr size_t N = 1024 * 2048 / 4 * 3;
    auto weights =
        wrs::pmr::generate_weights<T>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM, N, resource);

    auto prefixSum = wrs::reference::pmr::prefix_sum<T>(weights, resource);

    auto prefixError = wrs::test::pmr::assert_is_inclusive_prefix<T>(weights, prefixSum);
    if (prefixError) {
        SPDLOG_ERROR(prefixError.message());
        /* throw std::runtime_error("Test of test failed: wrs::reference::prefix_sum or " */
        /*                          "wrs::test::assert_is_inclusive_prefix is wrong"); */
    }

    { // Test floats against doubles
        auto weights = wrs::pmr::generate_weights<float>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
                                                         N, resource);

        std::pmr::vector<long double> doubleWeights{weights.begin(), weights.end(), resource};

        auto prefixSumFloat = wrs::reference::pmr::prefix_sum<float>(weights, resource);
        auto prefixSumDouble =
            wrs::reference::pmr::prefix_sum<long double>(doubleWeights, resource);
        assert(prefixSumFloat.size() == prefixSumDouble.size());
    }
}

static void testReduceReference(std::pmr::memory_resource* resource) {
    // ======== Tree reductions ============
    {
        constexpr size_t ELEM_COUNT = 1e4;
        auto elements =
            wrs::pmr::generate_weights<float>(wrs::Distribution::UNIFORM, ELEM_COUNT, resource);

        auto reduction = wrs::reference::pmr::tree_reduction<float>(elements, resource);
        if (std::abs(reduction - static_cast<float>(ELEM_COUNT)) > 0.01) {
            throw std::runtime_error("Test of test failed: wrs::reference::tree_reduce is wrong! "
                                     "Failed against uniform set");
        }
    }

    {
        constexpr size_t ELEM_COUNT = 100;
        auto elements = wrs::pmr::generate_weights<float>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
                                                          ELEM_COUNT, resource);

        auto reduction = wrs::reference::pmr::tree_reduction<float>(elements, resource);
        auto stdReduction =
            std::accumulate(elements.begin(), elements.end(), 0.0f, std::plus<float>());
        if (std::abs(reduction - stdReduction) > 0.01) {
            throw std::runtime_error(fmt::format(
                "Test of test failed: wrs::reference::tree_reduce is wrong!. Failed against std\n"
                "std got {}, tree_reduce got {}",
                stdReduction, reduction));
        }
    }
    // ======== Block reductions ============
    {
        constexpr size_t ELEM_COUNT = 1e4;
        auto elements =
            wrs::pmr::generate_weights<float>(wrs::Distribution::UNIFORM, ELEM_COUNT, resource);

        auto reduction = wrs::reference::pmr::block_reduction<float>(elements, 128, resource);
        if (std::abs(reduction - static_cast<float>(ELEM_COUNT)) > 0.01) {
            throw std::runtime_error(
                fmt::format("Test of test failed: wrs::reference::block_reduce is wrong!\nExpected "
                            "= {}, Got = {}",
                            ELEM_COUNT, reduction));
        }
    }
    {
        constexpr size_t ELEM_COUNT = 100;
        auto elements = wrs::pmr::generate_weights<float>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
                                                          ELEM_COUNT, resource);

        auto reduction = wrs::reference::pmr::block_reduction<float>(elements, 32, resource);
        auto stdReduction =
            std::accumulate(elements.begin(), elements.end(), 0.0f, std::plus<float>());
        if (std::abs(reduction - stdReduction) > 0.01) {
            throw std::runtime_error(
                fmt::format("Test of test failed: wrs::block_reduce is wrong! Failed against std\n"
                            "std got {}, block_reduce got {}",
                            stdReduction, reduction));
        }
    }

    // ========== Comparing numerical stability ==========
    {
        constexpr size_t ELEM_COUNT = 1e7;
        auto elements =
            wrs::pmr::generate_weights<float>(wrs::Distribution::UNIFORM, ELEM_COUNT, resource);

        auto reduction = wrs::reference::pmr::tree_reduction<float>(elements, resource);
        if (std::abs(reduction - static_cast<float>(ELEM_COUNT)) > 0.01) {
            SPDLOG_WARN(
                fmt::format("Numerical instability of tree reduction over {} uniform values: {}",
                            static_cast<float>(ELEM_COUNT),
                            std::abs(reduction - static_cast<float>(ELEM_COUNT))));
        }
    }
    /* for (size_t blockSize = 2; blockSize < 512; blockSize++) { */
    /* SPDLOG_DEBUG(fmt::format("blockSize = {}", blockSize)); */
    constexpr std::size_t blockSize = 128;
    constexpr size_t ELEM_COUNT = 1e7;
    auto elements =
        wrs::pmr::generate_weights<float>(wrs::Distribution::UNIFORM, ELEM_COUNT, resource);

    auto reduction = wrs::reference::pmr::block_reduction<float>(elements, blockSize, resource);
    if (std::abs(reduction - static_cast<float>(ELEM_COUNT)) > 0.01) {
        SPDLOG_WARN(fmt::format("Numerical instability of block reduction over {} uniform "
                                "values with blockSize = {} is {}",
                                static_cast<float>(ELEM_COUNT), blockSize,
                                std::abs(reduction - static_cast<float>(ELEM_COUNT))));
    }
    /* } */
}

static void testSplitTests(std::pmr::memory_resource* resource) {

    size_t N = 1024 * 2048;
    size_t K = N / 32;
    std::pmr::vector<float> weights =
        wrs::pmr::generate_weights<float>(wrs::Distribution::PSEUDO_RANDOM_UNIFORM, N, resource);

    float reduction = wrs::reference::tree_reduction<float>(weights);
    float average = reduction / static_cast<float>(weights.size());

    auto heavyLightPartition =
        wrs::reference::pmr::stable_partition<float>(weights, average, resource);
    const auto heavy = heavyLightPartition.heavy();
    const auto light = heavyLightPartition.light();

    std::pmr::vector<float> heavyPrefix = wrs::reference::pmr::prefix_sum<float>(heavy, resource);
    std::pmr::vector<float> lightPrefix = wrs::reference::pmr::prefix_sum<float>(light, resource);

    std::pmr::vector<wrs::Split<float, uint32_t>> splits =
        wrs::reference::pmr::splitK<float, uint32_t>(heavyPrefix, lightPrefix, average, N, K,
                                                     resource);

    auto err = wrs::test::pmr::assert_is_split<float, uint32_t>(splits, K, heavyPrefix, lightPrefix,
                                                                average, 0.01, resource);
    if (err) {
        SPDLOG_ERROR(fmt::format("Test of tests failed: split of it's assertion are invalid\n{}",
                                 err.message()));
    }

    /* wrs::reference::splitK(const std::span<T> heavyPrefix, const std::span<T> lightPrefix, T
     * mean, size_t N, size_t K) */
}

static void testAliasTableTest(std::pmr::memory_resource* resource) {

    // ============= SWEEPING-ALIAS-TABLE-CONSTRUCTION ===========
    {
        // NOTE: Sweeping alias table construction is not very stable =^(
        // It continously accumulates and error which we then find in the last element.
        // TODO: There are probably a bunch of ways how this can be improved one change
        // that is not in the original implementation that we did here is
        // that we normalize the alias table meaning that the redirection probablities
        // are not dependent on the original weights, this however might be one
        // point why we accumulate the error, because we work with pretty small numbers.
        // The original paper stores probablities relative to the average weight
        // and then removes the bias during sampling. For better abstraction we
        // have to reduce all alias table constructions into a normalized form,
        // which may be one point where we introduce the instability, because the original
        // paper does not show any instabilities that these ranges. They might however also be
        // using doubles instead of floats (Not sure about that)
        size_t N = 10;
        std::pmr::vector<float> weights = wrs::pmr::generate_weights<float>(
            wrs::Distribution::PSEUDO_RANDOM_UNIFORM, N, resource);
        float totalWeight = wrs::reference::pmr::tree_reduction<float>(weights, resource);

        SPDLOG_DEBUG("Compute reference");
        const wrs::pmr::AliasTable<float, uint32_t> aliasTable =
            wrs::reference::pmr::sweeping_alias_table<float, float, uint32_t>(weights, totalWeight,
                                                                              resource);
        const auto err = wrs::test::pmr::assert_is_alias_table<float, float, uint32_t>(
            weights, aliasTable, totalWeight, 1e-4, resource);
        if (err) {
            SPDLOG_ERROR(fmt::format("Test of tests failed: sweeping alias table references or "
                                     "assertions are invalid.\n{}",
                                     err.message()));
        }
    }
    { // Test psa reference construction
        const uint32_t N = 1024 * 2048;
        const uint32_t K = N / 32;
        SPDLOG_DEBUG(
            fmt::format("Testing wrs::reference::psa_alias_table... N = {}, K = {}", N, K));
        using weight_t = double;
        const std::pmr::vector<weight_t> weights = wrs::pmr::generate_weights<weight_t>(
            wrs::Distribution::PSEUDO_RANDOM_UNIFORM, N, resource);
        /* std::sort(weights.begin(), weights.end()); */
        /* std::vector<weight_t> weights = {2,2,2,2,2,1,1,1,1,1}; */
        assert(N == static_cast<uint32_t>(weights.size()));
        const float totalWeight = wrs::reference::kahan_reduction<weight_t>(weights);
        /* const float averageWeight = totalWeight / static_cast<float>(N); */

        wrs::pmr::AliasTable<weight_t, uint32_t> aliasTable =
            wrs::reference::pmr::psa_alias_table<weight_t, weight_t, uint32_t>(weights, K,

                                                                               resource);

        /*const auto err = wrs::test::pmr::assert_is_alias_table<float, float, uint32_t>(*/
        /*    weights, aliasTable, totalWeight, 1e-4, resource);*/

        const auto err = wrs::test::pmr::assert_is_alias_table<weight_t, weight_t, uint32_t>(
            weights, aliasTable, totalWeight, 1e-2, resource);
        if (err) {
            SPDLOG_ERROR(fmt::format(
                "Test of tests failed: psa alias table references or assertions are invalid.\n{}",
                err.message()));
        }
    }
}

static void testChiSquared(std::pmr::memory_resource* resource) {
    using I = std::size_t;
    using T = float;
    constexpr size_t N = static_cast<std::size_t>(10);
    constexpr size_t S = static_cast<std::size_t>(1e4);

    std::mt19937 rng{};
    std::pmr::vector<I> weights{N, resource};
    std::uniform_int_distribution<I> weightDist{1, 2};
    for (size_t i = 0; i < N; ++i) {
        weights[i] = weightDist(rng);
    }

    std::discrete_distribution<std::size_t> dist{weights.begin(), weights.end()};
    std::pmr::vector<I> samples{S, resource};
    for (size_t i = 0; i < S; ++i) {
        samples[i] = dist(rng);
    }

    std::pmr::vector<T> floatWeights{N, resource};
    for (size_t i = 0; i < N; ++i) {
        floatWeights[i] = std::max(static_cast<T>(weights[i]), T{});
    }

    T chiSquared = wrs::eval::chi_square<T, I>(samples, floatWeights);
    T critial = wrs::eval::chi_square_critical_value<T>(N, 0.05);
    if (chiSquared > critial) {
        SPDLOG_ERROR("Test of tests failed: chi square reference test failed");
    }
}

void wrs::test::testTests() {
    SPDLOG_INFO("Testing tests...");
    wrs::memory::StackResource stackResource{10000 * sizeof(float)};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource resource{&fallbackResource};

    SPDLOG_INFO("Testing reduce reference");
    stackResource.reset();
    // testReduceReference(&resource);

    SPDLOG_INFO("Testing partition assertions tests");
    stackResource.reset();
    // testPartitionTests(&resource);

    SPDLOG_INFO("Testing inclusive prefix assertion tests");
    stackResource.reset();
    // testPrefixTests(&resource);

    SPDLOG_INFO("Testing split assertion tests");
    stackResource.reset();
    // testSplitTests(&resource);

    SPDLOG_INFO("Testing alias table assertion tests");
    stackResource.reset();
    testAliasTableTest(&resource);

    SPDLOG_INFO("Testing chi squared evaluation");
    stackResource.reset();
    // testChiSquared(&resource);

    SPDLOG_INFO("Tested tests and references successfully!");
}
