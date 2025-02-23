#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/mean/Mean.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/device/statistics/chi_square/ChiSquare.hpp"
#include "src/device/wrs/alias/psa/PSA.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/host/assert/is_alias_table.hpp"
#include "src/host/assert/test.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/memory/FallbackResource.hpp"
#include "src/host/memory/SafeResource.hpp"
#include "src/host/memory/StackResource.hpp"
#include "src/host/reference/inverse_alias_table.hpp"
#include "src/host/statistics/js_divergence.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <spdlog/spdlog.h>

#include "src/host/reference/reduce.hpp"

namespace device::test::psa {

using Algorithm = PSA;
using Buffers = Algorithm::Buffers;
using Config = Algorithm::Config;
using weight_type = PSA::weight_type;

struct TestCase {
    Config config;
    host::glsl::uint N;
    host::Distribution distribution;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    /* TestCase{ */
    /*     .config = */
    /*         PSAConfig(AtomicMeanConfig(512, 8), */
    /*                   DecoupledPrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 32), */
    /*                   SerialSplitPackConfig(ScalarSplitConfig(2), ScalarPackConfig(2)), */
    /*                   true), */
    /*     .N = (1 << 28), */
    /*     .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM, */
    /*     .iterations = 5, */
    /* }, */
    /* TestCase{ */
    /*     .config = */
    /*         PSAConfig(AtomicMeanConfig(512, 8), */
    /*                   DecoupledPrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 32), */
    /*                   SerialSplitPackConfig(ScalarSplitConfig(16), SubgroupPackConfig(16, 4)), */
    /*                   true), */
    /*     .N = (1 << 28), */
    /*     .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM, */
    /*     .iterations = 5, */
    /* }, */
    TestCase{
        .config =
            PSAConfig(AtomicMeanConfig(512, 8),
                      DecoupledPrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 32),
                      InlineSplitPackConfig(2),
                      true),
        .N = (1 << 28),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 5,
    },
};

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> weights) {
    Buffers::WeightsView stageView{stage.weights, weights.size()};
    Buffers::WeightsView localView{buffers.weights, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void downloadToStage(const merian::CommandBufferHandle& cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            host::glsl::uint N) {

    {
        Buffers::MeanView stageView{stage.m_mean};
        Buffers::MeanView localView{buffers.m_mean};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::HeavyCountView stageView{stage.m_heavyCount};
        Buffers::HeavyCountView localView{buffers.m_heavyCount};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }

    {
        Buffers::AliasTableView stageView{stage.aliasTable, N};
        Buffers::AliasTableView localView{buffers.aliasTable, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results {
    float mean;
    host::glsl::uint heavyCount;
    std::pmr::vector<host::AliasTableEntry<weight_type, host::glsl::uint>> aliasTable;
};
static Results
downloadFromStage(Buffers& stage, host::glsl::uint N, std::pmr::memory_resource* resource) {
    Buffers::MeanView meanView{stage.m_mean};
    auto mean = meanView.download<weight_type>();

    Buffers::HeavyCountView heavyCountView{stage.m_heavyCount};
    auto heavyCount = heavyCountView.download<host::glsl::uint>();

    auto aliasTable = device::details::downloadAliasTableFromBuffer(stage.aliasTable, N, resource);

    return Results{
        .mean = mean,
        .heavyCount = heavyCount,
        .aliasTable = std::move(aliasTable),
    };
};

static bool runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        std::pmr::memory_resource* resource) {
    Buffers buffers = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                        testCase.config, testCase.N, true);
    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      testCase.config, testCase.N, true);

    std::string testName = fmt::format("{{{},N={}}}", testCase.config.name(), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    bool failed = false;
    double averageJSDivergence = 0;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(context.profiler, testName);
        context.queue->wait_idle();
        if (testCase.iterations > 1) {
            if (testCase.N > 1e6) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            }
        }

        // 1. Generate input
        context.profiler->start("Generate test input");
        const auto weights =
            host::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
        context.profiler->end();

        // 2. Begin recoding
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload test case indices
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload test case");
            SPDLOG_DEBUG("Uploading test case...");
            uploadTestCase(cmd, buffers, stage, weights);
        }

        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "PSA");
            SPDLOG_DEBUG("Building WRS");
            kernel.run(cmd, buffers, testCase.N, context.profiler);
        }

        // Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N);
        }

        // Submit to device
        context.profiler->end();
        context.profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd->end();
        context.queue->submit_wait(cmd);

        // Download from stage
        context.profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        [[maybe_unused]] Results results = downloadFromStage(stage, testCase.N, resource);
        context.profiler->end();

        // Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");

            if (testCase.N <= 1024) {

                fmt::println("ALIAS-TABLE:");
                for (std::size_t i = 0; i < results.aliasTable.size(); ++i) {
                    fmt::println("[{}]: ({},{})", i, results.aliasTable[i].p,
                                 results.aliasTable[i].a);
                }
                fmt::println("Mean: {}", results.mean);
                fmt::println("HeavyCount: {}", results.heavyCount);
            }

            const auto totalWeight = host::reference::reduce<weight_type>(weights);

            auto normalizedAliasTableWeights =
                host::reference::alias_table_to_normalized_weights<weight_type, host::glsl::uint>(
                    results.aliasTable);
            auto normalizedWeights = host::reference::normalize_weights<weight_type>(weights);

            /* auto r1 = host::reference::reduce<weight_type>(normalizedAliasTableWeights); */
            /* auto r2 = host::reference::reduce<weight_type>(normalizedWeights); */
            /*  */
            /* fmt::println("r1 = {}, r2 = {}, total={}", r1, r2, totalWeight); */

            auto jsDivergence = host::js_weight_divergence<weight_type>(normalizedAliasTableWeights,
                                                                        normalizedWeights);
            averageJSDivergence += jsDivergence;
            /* fmt::println("JS-Divergence: {}", jsDivergence); */

            if (testCase.N <= 1024) {
                const auto errAliasTable =
                    host::test::pmr::assert_is_alias_table<weight_type, weight_type,
                                                           host::glsl::uint>(
                        weights, results.aliasTable, totalWeight, 0.01, resource);
                if (errAliasTable) {
                    SPDLOG_ERROR("PSA-XXX constructs invalid alias table\n{}",
                                 errAliasTable.message());
                } else {
                    SPDLOG_INFO("PSA-XXX constructs correct alias table");
                }
            } else {
                SPDLOG_INFO("Skipping the tests of the alias table invariants");
            }
        }
        context.profiler->collect(true, true);
    }

    averageJSDivergence /= testCase.iterations;
    SPDLOG_INFO("JS-Divergence: {}", averageJSDivergence);

    if (averageJSDivergence > 0.3) {
        SPDLOG_ERROR("{} - is sampling something completely different", testName);
    } else if (averageJSDivergence > 0.15) {
        SPDLOG_ERROR("{} displays a significant bias", testName);
    } else if (averageJSDivergence > 0.05) {
        SPDLOG_WARN("{} displays a moderate bias", testName);
    } else {
        SPDLOG_INFO("{} is does not show any significant bias", testName);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing TODO algorithm");

    const host::test::TestContext testContext = host::test::setupTestContext(context);

    host::memory::StackResource stackResource{4096 * 2048};
    host::memory::FallbackResource fallbackResource{&stackResource};
    host::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    ChiSquare chiSquare{context, testContext.shaderCompiler};

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, resource);
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("All tests passed");
    } else {
        SPDLOG_ERROR(fmt::format("Failed {} out of {} tests", failCount,
                                 sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}

} // namespace device::test::psa
