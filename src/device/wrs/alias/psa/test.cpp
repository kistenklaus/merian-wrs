#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/mean/Mean.hpp"
#include "src/device/statistics/chi_square/ChiSquare.hpp"
#include "src/device/wrs/alias/psa/PSA.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/device/wrs/alias/psa/pack/subgroup/SubgroupPack.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPack.hpp"
#include "src/host/assert/is_alias_table.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/reference/inverse_alias_table.hpp"
#include "src/host/reference/partition.hpp"
#include "src/host/reference/prefix_sum.hpp"
#include "src/host/statistics/js_divergence.hpp"
#include "src/host/test/context.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <spdlog/spdlog.h>

#include "src/host/reference/reduce.hpp"
#include "vulkan/vulkan_enums.hpp"

#ifndef MERIAN_PROFILER_ENABLE
#define MERIAN_PROFILER_ENABLE
#endif

namespace device::psa {

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

static const TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            SerialSplitPackConfig(ScalarSplitConfig(16), SubgroupPackConfig(16, 8)),
                            false),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            SerialSplitPackConfig(ScalarSplitConfig(16), SubgroupPackConfig(16, 8)),
                            false),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            SerialSplitPackConfig(ScalarSplitConfig(16), SubgroupPackConfig(16, 8)),
                            false),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            SerialSplitPackConfig(ScalarSplitConfig(16), SubgroupPackConfig(16, 8)),
                            false),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            SerialSplitPackConfig(ScalarSplitConfig(16), SubgroupPackConfig(16, 8)),
                            false),
        .N = static_cast<uint32_t>(1e8),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(16, 8, 512),
                            false),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(16, 8, 512),
                            false),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(16, 8, 512),
                            false),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(16, 8, 512),
                            false),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(16, 8, 512),
                            false),
        .N = static_cast<uint32_t>(1e8),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(2, 32, 512),
                            false),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(2, 32, 512),
                            false),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(2, 32, 512),
                            false),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(2, 32, 512),
                            false),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = PSAConfig(AtomicMeanConfig(),
                            DecoupledPrefixPartitionConfig(),
                            InlineSplitPackConfig(2, 32, 512),
                            false),
        .N = static_cast<uint32_t>(1e8),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
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

static void runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        std::pmr::memory_resource* resource) {
    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context.context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context.context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(context.queue);
    Buffers buffers = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                        testCase.config, testCase.N, true);
    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      testCase.config, testCase.N, true);

    std::string testName = fmt::format("{{{},N={}}}", testCase.config.name(), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    host::test::TestResultType out = host::test::SUCCESS;
    double averageJSDivergence = 0;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(profiler, testName);
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
        profiler->start("Generate test input");
        const auto weights =
            host::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
        profiler->end();

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();

        // 2. Begin recoding
        /* std::string recordingLabel = fmt::format("Recording : {}", testName); */
        /* context.profiler->start(recordingLabel); */
        /* context.profiler->cmd_start(cmd, recordingLabel); */

        // 3. Upload test case indices
        {
            {
                /* MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload test case"); */
                SPDLOG_DEBUG("Uploading test case...");
                uploadTestCase(cmd, buffers, stage, weights);
            }
        }

        {
            {
                /* MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "PSA"); */

                profiler->start("PSA");
                profiler->cmd_start(cmd, "PSA");

                kernel.run(cmd, buffers, testCase.N, profiler);

                profiler->end();
                profiler->cmd_end(cmd);
            }
        }

        // Submit to device

        // Download results to stage
        {

            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eTransfer,
                         buffers.weights->buffer_barrier(vk::AccessFlagBits::eShaderRead,
                                                         vk::AccessFlagBits::eTransferWrite));

            /* MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage"); */
            /* SPDLOG_DEBUG("Downloading results to stage..."); */
            downloadToStage(cmd, buffers, stage, testCase.N);
        }
        cmd->end();
        context.queue->submit_wait(cmd);

        // Download from stage
        profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results results = downloadFromStage(stage, testCase.N, resource);

        profiler->end();

        // Test results
        {
            MERIAN_PROFILE_SCOPE(profiler, "Testing results");

            if ((testCase.N <= 1024)) {

                auto totalWeight = host::reference::reduce<float>(weights);
                auto part =
                    host::reference::stable_partition<float>(weights, totalWeight / weights.size());
                auto lightPrefix = host::reference::prefix_sum<float>(part.light());
                auto heavyPrefix = host::reference::prefix_sum<float>(part.heavy());

                /*fmt::println("LIGHT");*/
                /*for (std::size_t i = 0; i < lightPrefix.size(); ++i) {*/
                /*  fmt::println("[{}]: {}         ({})", i, lightPrefix[i], part.light()[i]);*/
                /*}*/
                /*fmt::println("HEAVY");*/
                /*for (std::size_t i = 0; i < heavyPrefix.size(); ++i) {*/
                /*  fmt::println("[{}]: {}         ({})", i, heavyPrefix[i], part.heavy()[i]);*/
                /*}*/

                /*fmt::println("ALIAS-TABLE:");*/
                /*for (std::size_t i = 0; i < results.aliasTable.size(); ++i) {*/
                /*    fmt::println("[{:>3}]: ({:.4f},{:>3})", i, results.aliasTable[i].p,*/
                /*                 results.aliasTable[i].a);*/
                /*}*/

                auto normalizedWeight = host::reference::normalize_weights<float>(weights);
                auto sampledWeights =
                    host::reference::alias_table_to_normalized_weights<float, host::glsl::uint>(
                        results.aliasTable);
                fmt::println("ALIAS-TABLE:");
                for (std::size_t i = 0; i < results.aliasTable.size(); ++i) {
                    fmt::println("[{:>3}]: ({:.4f},{:>3}) :: {:.4f}  ->  {:.4f}    ({:.3f})", i,
                                 results.aliasTable[i].p, results.aliasTable[i].a,
                                 normalizedWeight[i], sampledWeights[i], weights[i]);
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

            if (testCase.N <= 1024 || true) {
                const auto errAliasTable =
                    host::test::pmr::assert_is_alias_table<weight_type, weight_type,
                                                           host::glsl::uint>(
                        weights, results.aliasTable, totalWeight, 1, resource);
                if (errAliasTable) {
                    SPDLOG_WARN("{} Numerical instabilities during PSA construction. (Check "
                                "JS-Divergence!)\n{}",
                                testName, errAliasTable.message());
                    out += host::test::WARNING;
                } else {
                    SPDLOG_INFO("PSA-XXX constructs correct alias table");
                }
            } else {
                SPDLOG_INFO("Skipping the tests of the alias table invariants");
            }
        }
        profiler->collect(true, true);
    }

    averageJSDivergence /= testCase.iterations;
    SPDLOG_INFO("JS-Divergence: {}", averageJSDivergence);

    if (averageJSDivergence > 0.3) {
        SPDLOG_ERROR("{} - is sampling something completely different", testName);
        out += host::test::ERROR;
    } else if (averageJSDivergence > 0.15) {
        SPDLOG_ERROR("{} displays a significant bias", testName);
        out += host::test::ERROR;
    } else if (averageJSDivergence > 0.05) {
        SPDLOG_WARN("{} displays a moderate bias", testName);
        out += host::test::WARNING;
    } else {
        SPDLOG_INFO("{} is does not show any significant bias", testName);
    }

    const auto report = profiler->get_report().gpu_report;
    auto entry = std::ranges::find_if(
        report, [&](const merian::Profiler::ReportEntry& entry) { return entry.name == "PSA"; });
    if (entry == report.end()) {
        fmt::println("ENTRIES:");
        for (const auto& x : report) {
            fmt::println("{}", x.name);
        }
        throw std::runtime_error("Impossible state 1");
    }

    auto T = splitPackConfigInvocPerPack(context.context, testCase.config.splitPackConfig);
    auto S = splitPackConfigSplitSize(testCase.config.splitPackConfig);
    context.pushResult(
        "PSA", testCase.config.className(), out, entry->duration, entry->std_deviation,
        {host::test::TestProperty{.name = "N",
                                  .value = std::format("{:.0e}", static_cast<float>(testCase.N))},
         host::test::TestProperty{
             .name = "T",
             .value = fmt::format("{}", T),
         },
         host::test::TestProperty{
             .name = "split-size",
             .value = fmt::format("{}", S),
         }});
}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing PSA algorithm");

    std::pmr::memory_resource* resource = context.memory_resource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, testCase, resource);
    }
}

} // namespace device::psa
