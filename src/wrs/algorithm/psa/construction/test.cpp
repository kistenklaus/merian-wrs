#include "./test.hpp"

#include "src/wrs/gen/weight_generator.h"
#include <fmt/base.h>
#include <src/wrs/memory/FallbackResource.hpp>
#include <src/wrs/memory/SafeResource.hpp>
#include <src/wrs/memory/StackResource.hpp>
#include <src/wrs/reference/reduce.hpp>
#include <src/wrs/test/is_alias_table.hpp>
#include <src/wrs/test/test.hpp>
#include <src/wrs/types/alias_table.hpp>
#include <src/wrs/types/partition.hpp>

#include "./PSAC.hpp"
#include "src/wrs/types/split.hpp"

using Buffers = wrs::PSAC::Buffers;
using weight_type = wrs::PSAC::weight_t;

namespace wrs::test::psac {

struct TestCase {
    std::size_t weightCount;
    Distribution distribution;

    PSACConfig config;

    std::size_t iterations;
};

constexpr TestCase TEST_CASES[] = {
    TestCase{
        .weightCount = static_cast<std::size_t>(1e7),
        .distribution = Distribution::PSEUDO_RANDOM_UNIFORM,
        .config = {},
        .iterations = 1,
    },
};

static void uploadWeights(const merian::CommandBufferHandle& cmd,
                          const Buffers& buffers,
                          const Buffers& stage,
                          const std::span<const weight_type> weights) {
    Buffers::WeightsView stageView{stage.weights, weights.size()};
    Buffers::WeightsView localView{buffers.weights, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void zeroDecoupledStates(const merian::CommandBufferHandle cmd,
                                const Buffers& buffers,
                                const std::size_t weightCount,
                                std::size_t prefixPartitionSize) {
    const std::size_t prefixWorkgroupCount =
        (weightCount + prefixPartitionSize - 1) / prefixPartitionSize;
    Buffers::PartitionDecoupledStateView partitionStateView{buffers.partitionDecoupledState,
                                                            prefixWorkgroupCount};
    partitionStateView.zero(cmd);
    partitionStateView.expectComputeRead(cmd);
}

static void downloadAliasTableToStage(const merian::CommandBufferHandle cmd,
                                      const Buffers& buffers,
                                      const Buffers& stage,
                                      const std::size_t weightCount) {
    Buffers::AliasTableView stageView{stage.aliasTable, weightCount};
    Buffers::AliasTableView localView{buffers.aliasTable, weightCount};
    localView.expectComputeWrite();
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

static void downloadSplitsToStage(const merian::CommandBufferHandle cmd,
                                  const Buffers& buffers,
                                  const Buffers& stage,
                                  const std::size_t splitCount) {
    Buffers::SplitsView stageView{stage.splits, splitCount};
    Buffers::SplitsView localView{buffers.splits, splitCount};
    localView.expectComputeWrite();
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

wrs::pmr::AliasTable<weight_type, wrs::glsl::uint> downloadAliasTableFromStage(
    const Buffers& stage, const std::size_t weightCount, std::pmr::memory_resource* resource);

std::pmr::vector<wrs::Split<float, glsl::uint>> downloadSplitsFromStage(
    const Buffers& stage, const std::size_t splitCount, std::pmr::memory_resource* resource);

static bool runTestCase(const wrs::test::TestContext& context,
                        const Buffers& buffers,
                        const Buffers& stage,
                        std::pmr::memory_resource* resource,
                        const TestCase& testCase) {
    std::string testName = fmt::format(
        "{{weightCount={},dist={},splitSize={}}}", testCase.weightCount,
        wrs::distribution_to_pretty_string(testCase.distribution), testCase.config.splitSize);
    SPDLOG_INFO("Running test case:{}", testName);

    SPDLOG_DEBUG("Creating ScalarPsa instance");
    wrs::PSAC psac{context.context, context.shaderCompiler, testCase.config};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(context.profiler, testName);
        context.queue->wait_idle();
        if (testCase.iterations > 1) {
            if (testCase.weightCount > static_cast<std::size_t>(1e6)) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            }
        }
        const std::size_t N = testCase.weightCount;

        std::size_t splitSize = testCase.config.splitSize;
        std::size_t splitCount = (testCase.weightCount + splitSize - 1) / splitSize;

        // 1. Generate weights
        std::pmr::vector<weight_type> weights{resource};
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            SPDLOG_DEBUG("Generating weights...");
            weights = wrs::pmr::generate_weights<weight_type>(testCase.distribution, N, resource);
        }

        // 2. Begin recoding
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);

#ifdef MERIAN_PROFILER_ENABLE
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);
#endif

        // 3.0 Upload weights
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload weights indices");
            SPDLOG_DEBUG("Uploading weights...");
            uploadWeights(cmd, buffers, stage, weights);

            const std::size_t prefixPartitionSize = psac.getPrefixPartitionSize();
            zeroDecoupledStates(cmd, buffers, N, prefixPartitionSize);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            psac.run(cmd, buffers, N, context.profiler);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadAliasTableToStage(cmd, buffers, stage, N);
            downloadSplitsToStage(cmd, buffers, stage, splitCount);
        }
        // 6. Submit to device
        {
#ifdef MERIAN_PROFILER_ENABLE
            context.profiler->end();
            context.profiler->cmd_end(cmd);
            SPDLOG_DEBUG("Submitting to device...");
#endif
            cmd->end();
            context.queue->submit_wait(cmd);
        }
        // 7. Download from stage
        wrs::pmr::AliasTable<weight_type, wrs::glsl::uint> aliasTable{resource};
        std::pmr::vector<wrs::Split<float, glsl::uint>> splits{resource};
        {
            SPDLOG_DEBUG("Downloading results from stage...");
            aliasTable = downloadAliasTableFromStage(stage, N, resource);
            splits = downloadSplitsFromStage(stage, splitCount, resource);
        }

        // 8. Test pack invariants
        {
            SPDLOG_DEBUG("Computing reference totalWeight with the kahan sum algorithm");
            const auto totalWeight = wrs::reference::kahan_reduction<weight_type>(weights);
            SPDLOG_DEBUG("Testing results");
            const auto err =
                wrs::test::pmr::assert_is_alias_table<weight_type, weight_type, wrs::glsl::uint>(
                    weights, aliasTable, totalWeight, 0.01, resource);
            if (err) {
                SPDLOG_ERROR(err.message());

                if (testCase.weightCount < 1024) {
                    fmt::println("SPLITS");
                    for (std::size_t i = 0; i < splits.size(); ++i) {
                        fmt::println("[{}]: ({},{},{})", i, splits[i].i, splits[i].j,
                                     splits[i].spill);
                    }
                    /*for (std::size_t i = 0; i < aliasTable.size(); ++i) {*/
                    /*    fmt::println("[{}]: ({},{})", i, aliasTable[i].p, aliasTable[i].a);*/
                    /*}*/
                }
            }
            /* std::vector<glsl::uint> samples = */
            /*     wrs::reference::sample_alias_table<float, glsl::uint>(aliasTable, 1e9); */
            /* float rmse = wrs::eval::rmse<float, glsl::uint>(weights, samples); */
            /* fmt::println("S = {}, RMSE = {}", 1e9, rmse); */
        }

        context.profiler->collect(true, true);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing parallel split alias table construction algorithm");

    TestContext c = setupTestContext(context);

    std::size_t maxWeightCount = 0;
    std::size_t maxSplitCount = 0;
    glsl::uint maxPrefixPartitionSize = 0;
    glsl::uint maxMeanPartitionSize = 0;
    for (const auto& testCase : TEST_CASES) {
        maxWeightCount = std::max(maxWeightCount, testCase.weightCount);

        std::size_t splitSize = testCase.config.splitSize;
        std::size_t splitCount = (testCase.weightCount + splitSize - 1) / splitSize;
        maxSplitCount = std::max(maxSplitCount, splitCount);
        maxPrefixPartitionSize =
            std::max(maxPrefixPartitionSize, testCase.config.prefixPartitionConfig.partitionSize());
        maxMeanPartitionSize =
            std::max(maxMeanPartitionSize, testCase.config.meanConfig.partitionSize());
    }

    auto buffers =
        Buffers::allocate(c.alloc, maxWeightCount, maxMeanPartitionSize, maxPrefixPartitionSize,
                          maxSplitCount, merian::MemoryMappingType::NONE);
    auto stage =
        Buffers::allocate(c.alloc, maxWeightCount, maxMeanPartitionSize, maxPrefixPartitionSize,
                          maxSplitCount, merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    memory::StackResource stackResource{buffers.weights->get_size() * 10};
    memory::FallbackResource fallbackResource{&stackResource};
    memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        if (runTestCase(c, buffers, stage, resource, testCase)) {
            failCount += 1;
        }
        stackResource.reset();
    }
    c.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(c.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("parallel split alias table construction passed all tests");
    } else {
        SPDLOG_ERROR(fmt::format(
            "parallel split alias table construction algorithm failed {} out of {} tests",
            failCount, sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}

} // namespace wrs::test::psac
