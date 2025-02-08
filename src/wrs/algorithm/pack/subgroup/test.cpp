#include "src/wrs/test/test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/pack/scalar/ScalarPack.hpp"
#include "src/wrs/algorithm/pack/subgroup/SubgroupPack.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_alias_table.hpp"
#include "src/wrs/test/is_split.hpp"
#include "src/wrs/types/alias_table.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>

using namespace wrs::test;

namespace wrs::test::subgroup_pack {

using Algorithm = SubgroupPack;
using Buffers = Algorithm::Buffers;

struct TestCase {
    SubgroupPackConfig config;
    glsl::uint weightCount; // N
    Distribution distribution;
    glsl::uint iterations; // K
};

constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = SubgroupPackConfig(16, 512, 8),
        .weightCount = 1024 * 2048,
        .distribution = Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 1,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    std::uint32_t maxWeightCount = 0;
    std::uint32_t maxSplitCount = 0;
    for (const auto& testCase : TEST_CASES) {
        maxWeightCount = std::max(maxWeightCount, testCase.weightCount);
        const glsl::uint splitCount =
            (testCase.weightCount + testCase.config.splitSize - 1) / testCase.config.splitSize;
        maxSplitCount = std::max(maxSplitCount, splitCount);
    }
    Buffers buffers = Buffers::allocate(context.alloc, maxWeightCount, maxSplitCount,
                                        merian::MemoryMappingType::NONE);
    Buffers stage = Buffers::allocate(context.alloc, maxWeightCount, maxSplitCount,
                                      merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    return std::make_tuple(buffers, stage);
}

static void uploadPartitionIndices(const vk::CommandBuffer cmd,
                                   const std::span<const wrs::glsl::uint> heavyIndices,
                                   const std::span<const wrs::glsl::uint> reverseLightIndices,
                                   const Buffers& buffers,
                                   const Buffers& stage,
                                   std::pmr::memory_resource* resource) {
    const std::size_t N = heavyIndices.size() + reverseLightIndices.size();
    wrs::Partition<wrs::glsl::uint, std::pmr::vector<wrs::glsl::uint>> partition{
        std::pmr::vector<wrs::glsl::uint>{N, resource},
        static_cast<std::ptrdiff_t>(heavyIndices.size())};
    std::memcpy(partition.heavy().data(), heavyIndices.data(), heavyIndices.size_bytes());
    std::memcpy(partition.light().data(), reverseLightIndices.data(),
                reverseLightIndices.size_bytes());

    Buffers::PartitionIndicesView stageView{stage.partitionIndices, N};
    Buffers::PartitionIndicesView localView{buffers.partitionIndices, N};
    stageView.attribute<"heavyCount">().template upload<wrs::glsl::uint>(heavyIndices.size());
    stageView.attribute<"heavyLightIndices">().template upload<wrs::glsl::uint>(
        partition.storage());
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void uploadPartition(vk::CommandBuffer cmd,
    std::span<const float> partition,
    Buffers& buffers,
    Buffers& stage) {
  Buffers::PartitionView stageView{stage.partition, partition.size()};
  Buffers::PartitionView localView{buffers.partition, partition.size()};
  stageView.upload(partition);
  stageView.copyTo(cmd, localView);
  localView.expectComputeRead(cmd);
}

static void uploadWeights(vk::CommandBuffer cmd,
                          std::span<const float> weights,
                          Buffers& buffers,
                          Buffers& stage) {
    Buffers::WeightsView stageView{stage.weights, weights.size()};
    Buffers::WeightsView localView{buffers.weights, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

void uploadSplits(const vk::CommandBuffer cmd,
                  std::span<const wrs::Split<float, wrs::glsl::uint>> splits,
                  Buffers& buffers,
                  Buffers& stage);

static void
uploadMean(vk::CommandBuffer cmd, float averageWeight, Buffers& buffers, Buffers& stage) {
    Buffers::MeanView stageView{stage.mean};
    Buffers::MeanView localView{buffers.mean};
    stageView.upload(averageWeight);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void downloadAliasTableToStage(const vk::CommandBuffer cmd,
                                      const std::size_t N,
                                      const Buffers& buffers,
                                      const Buffers& stage) {
    Buffers::AliasTableView stageView{stage.aliasTable, N};
    Buffers::AliasTableView localView{buffers.aliasTable, N};
    localView.expectComputeWrite();
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

wrs::pmr::AliasTable<float, wrs::glsl::uint>
downloadAliasTableFromStage(std::size_t N, Buffers& stage, std::pmr::memory_resource* resource);

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName = fmt::format(
        "{{weightCount={},dist={},splitSize={}}}", testCase.weightCount,
        wrs::distribution_to_pretty_string(testCase.distribution), testCase.config.splitSize);
    SPDLOG_INFO("Running test case:{}", testName);

    SPDLOG_DEBUG("Creating ScalarPack instance");
    Algorithm kernel{context.context, testCase.config};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(context.profiler, testName);
        context.queue->wait_idle();
        if (testCase.iterations > 1) {
            if (testCase.weightCount > 1e6) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            }
        }
        const std::size_t N = testCase.weightCount;

        const std::size_t K =
            (testCase.weightCount + testCase.config.splitSize - 1) / testCase.config.splitSize;

        // 1. Generate weights
        std::pmr::vector<float> weights{resource};
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            SPDLOG_DEBUG("Generating weights...");
            weights = wrs::pmr::generate_weights<float>(testCase.distribution, N, resource);
        }

        // 1.1 Compute reference input
        SPDLOG_DEBUG("Computing input from references...");
        float totalWeight;
        float averageWeight;
        wrs::Partition<wrs::glsl::uint, std::pmr::vector<wrs::glsl::uint>> heavyLightIndicies;
        std::pmr::vector<float> heavyPrefix{resource};
        std::pmr::vector<float> lightPrefix{resource};
        std::pmr::vector<wrs::Split<float, wrs::glsl::uint>> splits{resource};
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute reduction, partition,prefix,splits");
            SPDLOG_DEBUG("Running kahan reduction to compute totalWeight...");
            totalWeight = wrs::reference::kahan_reduction<float>(weights);
            averageWeight = totalWeight / static_cast<float>(N);
            SPDLOG_DEBUG("Running stable_partition_indicies to compute partition indices...");
            heavyLightIndicies = wrs::reference::pmr::stable_partition_indicies<float, uint32_t>(
                weights, averageWeight, resource);

            SPDLOG_DEBUG("Running prefix_sum to compute heavyPrefix...");
            const auto& deref = [&](const uint32_t i) -> float { return weights[i]; };
            heavyPrefix = wrs::reference::pmr::prefix_sum<float>(
                heavyLightIndicies.heavy() | std::views::transform(deref), resource);
            SPDLOG_DEBUG("Running prefix_sum to compute lightPrefix...");
            lightPrefix = wrs::reference::pmr::prefix_sum<float>(
                heavyLightIndicies.light() | std::views::transform(deref), resource);
            SPDLOG_DEBUG("Running splitK to compute splits...");
            splits = wrs::reference::pmr::splitK<float, wrs::glsl::uint>(
                heavyPrefix, lightPrefix, averageWeight, N, K, resource);
        }
        const auto heavyIndices = heavyLightIndicies.heavy();
        std::pmr::vector<wrs::glsl::uint> reverseLightIndices{
            heavyLightIndicies.light().begin(), heavyLightIndicies.light().end(), resource};
        std::ranges::reverse(reverseLightIndices);

        std::vector<float> partition(weights.size());
        for (std::size_t i = 0; i < heavyIndices.size(); ++i) {
          partition[i] = weights[heavyIndices[i]];
        }
        for (std::size_t i = 0; i < reverseLightIndices.size(); ++i) {
          partition[i + heavyIndices.size()] = weights[reverseLightIndices[i]];
        }

        /*auto err = wrs::test::pmr::assert_is_split<float, glsl::uint>(splits, K, heavyPrefix,
         * lightPrefix, averageWeight, 0.01, resource);*/
        /*if (err) {*/
        /*  SPDLOG_ERROR("Invalid reference split: \n{}", err.message());*/
        /**/
        /*}*/

        if (K <= 128) {
            fmt::println("Splits:");
            for (std::size_t i = 0; i < splits.size(); ++i) {
                fmt::println("[{}]: ({},{},{})", i, splits[i].i, splits[i].j, splits[i].spill);
            }
        }

        // 2. Begin recoding
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // 3.0 Upload partition indices
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload partition indices");
            SPDLOG_DEBUG("Uploading partition indices...");
            uploadPartitionIndices(cmd, heavyIndices, reverseLightIndices, buffers, stage,
                                   resource);
        }
        {
          uploadPartition(cmd, partition, buffers, stage);
        }
        // 3.1 Upload splits
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload splits");
            SPDLOG_DEBUG("Uploading splits...");
            uploadSplits(cmd, splits, buffers, stage);
        }
        { // 3.2 Upload mean
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload mean");
            SPDLOG_DEBUG("Uploading mean...");
            uploadMean(cmd, averageWeight, buffers, stage);
        }
        { // 3.3
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload weights");
            SPDLOG_DEBUG("Uploading weights...");
            uploadWeights(cmd, weights, buffers, stage);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.weightCount);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadAliasTableToStage(cmd, N, buffers, stage);
        }
        // 6. Submit to device
        {
            context.profiler->end();
            context.profiler->cmd_end(cmd);
            SPDLOG_DEBUG("Submitting to device...");
            cmd.end();
            context.queue->submit_wait(cmd);
        }
        // 7. Download from stage
        wrs::pmr::AliasTable<float, wrs::glsl::uint> aliasTable{resource};
        {
            SPDLOG_DEBUG("Downloading results from stage...");
            aliasTable = downloadAliasTableFromStage(N, stage, resource);
        }
        /* fmt::println("SPLITS"); */
        /* { */
        /*   for (size_t i = 0; i < splits.size(); ++i) { */
        /*     fmt::println("({},{},{})", splits[i].i, splits[i].j, splits[i].spill); */
        /*   } */
        /* } */
        if (testCase.weightCount <= 1024) {
            fmt::println("TABLE");
            // Print table
            for (size_t i = 0; i < aliasTable.size(); ++i) {
                fmt::println("{} = ({},{})", i, aliasTable[i].p, aliasTable[i].a);
            }
        }

        fmt::println("ENTRIES-DIRECTING-TO-1000");
        for (size_t i = 0; i < aliasTable.size(); ++i) {
            if (aliasTable[i].a == 1000) {
                fmt::println("{} = ({},{})", i, aliasTable[i].p, aliasTable[i].a);
            }
        }
        // 8. Test pack invariants
        {

            SPDLOG_DEBUG("Testing results");
            const auto err = pmr::assert_is_alias_table<float, float, wrs::glsl::uint>(
                weights, aliasTable, totalWeight, 0.01, resource);
            if (err) {
                SPDLOG_ERROR(err.message());
            }

            /*if (testCase.weightCount < 1024 || true) {*/
            /*    for (std::size_t i = 0; i < aliasTable.size(); ++i) {*/
            /*        fmt::println("[{}]: ({},{})", i, aliasTable[i].p, aliasTable[i].a);*/
            /*    }*/
            /*}*/
        }

        context.profiler->collect(true, true);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing subgroup pack algorithm");
    TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{buffers.aliasTable->get_size() * 10};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        bool failed = runTestCase(testContext, testCase, buffers, stage, resource);
        if (failed) {
            failCount += 1;
        }
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("Scalar pack algorithm passed all tests");
    } else {
        SPDLOG_ERROR(fmt::format("Scalar pack algorithm failed {} out of {} tests", failCount,
                                 sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}

} // namespace wrs::test::subgroup_pack
