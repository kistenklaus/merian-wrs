#include "./test.hpp"
#include "./test/test_cases.hpp"
#include "./test/test_setup.hpp"
#include "./test/test_types.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/pack/simd/SimdPack.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_alias_table.hpp"
#include "src/wrs/test/test.hpp"
#include "src/wrs/types/alias_table.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include "src/renderdoc.hpp"

#define MERIAN_PROFILER_ENABLE

using namespace wrs::test::simd_pack;
using namespace wrs::test;

vk::DeviceSize wrs::test::simd_pack::sizeOfWeight(const WeightType ty) {
    switch (ty) {
    case WEIGHT_TYPE_FLOAT:
        return sizeof(float);
    }
    throw std::runtime_error("NOT IMPLEMENTED");
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

template <wrs::arithmetic weight_t>
static void uploadWeights(vk::CommandBuffer cmd,
                          std::span<const weight_t> weights,
                          Buffers& buffers,
                          Buffers& stage) {
    Buffers::WeightsView stageView{stage.weights, weights.size()};
    Buffers::WeightsView localView{buffers.weights, weights.size()};
    stageView.template upload<weight_t>(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

template <wrs::arithmetic weight_t>
static void uploadSplits(vk::CommandBuffer cmd,
                         std::span<const wrs::Split<weight_t, wrs::glsl::uint>> splits,
                         Buffers& buffers,
                         Buffers& stage) {
    Buffers::SplitsView stageView{stage.splits, splits.size()};
    Buffers::SplitsView localView{buffers.splits, splits.size()};
    stageView.upload<wrs::Split<weight_t, wrs::glsl::uint>>(splits);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

template <wrs::arithmetic weight_t>
static void
uploadMean(vk::CommandBuffer cmd, weight_t averageWeight, Buffers& buffers, Buffers& stage) {
    Buffers::MeanView stageView{stage.mean};
    Buffers::MeanView localView{buffers.mean};
    stageView.template upload(averageWeight);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

template <wrs::arithmetic weight_t>
static void
downloadAliasTableToStage(vk::CommandBuffer cmd, std::size_t N, Buffers& buffers, Buffers& stage) {
    Buffers::AliasTableView stageView{stage.aliasTable, N};
    Buffers::AliasTableView localView{buffers.aliasTable, N};
    localView.expectComputeWrite();
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

template <wrs::arithmetic weight_t>
static wrs::pmr::AliasTable<weight_t, wrs::glsl::uint>
downloadAliasTableFromStage(std::size_t N, Buffers& stage, std::pmr::memory_resource* resource) {
    Buffers::AliasTableView stageView{stage.aliasTable, N};
    using Entry = wrs::AliasTableEntry<weight_t, wrs::glsl::uint>;

    return stageView.template download<Entry, wrs::pmr_alloc<Entry>>(resource);
}

template <wrs::arithmetic weight_t>
static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{workgroupSize={},weightCount={},dist={}splitCount={}}}", testCase.workgroupSize,testCase.weightCount,
                    wrs::distribution_to_pretty_string(testCase.distribution), testCase.splitCount);
    SPDLOG_INFO("Running test case:{}", testName);

    SPDLOG_DEBUG("Creating ScalarPack instance");
    wrs::SimdPack<weight_t> kernel{context.context, testCase.workgroupSize};

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
        const std::size_t K = testCase.splitCount;

        // 1. Generate weights
        std::pmr::vector<weight_t> weights{resource};
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            SPDLOG_DEBUG("Generating weights...");
            weights = wrs::pmr::generate_weights<weight_t>(testCase.distribution, N, resource);
        }

        // 1.1 Compute reference input
        SPDLOG_DEBUG("Computing input from references...");
        weight_t totalWeight;
        weight_t averageWeight;
        wrs::Partition<wrs::glsl::uint, std::pmr::vector<wrs::glsl::uint>> heavyLightIndicies;
        std::pmr::vector<weight_t> heavyPrefix{resource};
        std::pmr::vector<weight_t> lightPrefix{resource};
        std::pmr::vector<wrs::Split<weight_t, wrs::glsl::uint>> splits{resource};
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute reduction, partition,prefix,splits");
            SPDLOG_DEBUG("Running kahan reduction to compute totalWeight...");
            totalWeight = wrs::reference::kahan_reduction<weight_t>(weights);
            averageWeight = totalWeight / static_cast<weight_t>(N);
            SPDLOG_DEBUG("Running stable_partition_indicies to compute partition indices...");
            heavyLightIndicies = wrs::reference::pmr::stable_partition_indicies<weight_t, uint32_t>(
                weights, averageWeight, resource);

            SPDLOG_DEBUG("Running prefix_sum to compute heavyPrefix...");
            const auto& deref = [&](const uint32_t i) -> weight_t { return weights[i]; };
            heavyPrefix = wrs::reference::pmr::prefix_sum<weight_t>(
                heavyLightIndicies.heavy() | std::views::transform(deref), resource);
            SPDLOG_DEBUG("Running prefix_sum to compute lightPrefix...");
            lightPrefix = wrs::reference::pmr::prefix_sum<weight_t>(
                heavyLightIndicies.light() | std::views::transform(deref), resource);
            SPDLOG_DEBUG("Running splitK to compute splits...");
            splits = wrs::reference::pmr::splitK<weight_t, wrs::glsl::uint>(
                heavyPrefix, lightPrefix, averageWeight, N, K, resource);
        }
        const auto heavyIndices = heavyLightIndicies.heavy();
        std::pmr::vector<wrs::glsl::uint> reverseLightIndices{
            heavyLightIndicies.light().begin(), heavyLightIndicies.light().end(), resource};
        std::ranges::reverse(reverseLightIndices);

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
        // 3.1 Upload splits
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload splits");
            SPDLOG_DEBUG("Uploading splits...");
            uploadSplits<weight_t>(cmd, splits, buffers, stage);
        }
        { // 3.2 Upload mean
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload mean");
            SPDLOG_DEBUG("Uploading mean...");
            uploadMean<weight_t>(cmd, averageWeight, buffers, stage);
        }
        { // 3.3
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload weights");
            SPDLOG_DEBUG("Uploading weights...");
            uploadWeights<weight_t>(cmd, weights, buffers, stage);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, testCase.weightCount, testCase.splitCount, buffers);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadAliasTableToStage<weight_t>(cmd, N, buffers, stage);
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
        wrs::pmr::AliasTable<weight_t, wrs::glsl::uint> aliasTable{resource};
        {
            SPDLOG_DEBUG("Downloading results from stage...");
            aliasTable = downloadAliasTableFromStage<weight_t>(N, stage, resource);
        }
         fmt::println("SPLITS"); 
         { 
           /*for (size_t i = 0; i < splits.size(); ++i) { */
           /*  fmt::println("({},{},{})", splits[i].i, splits[i].j, splits[i].spill); */
           /*} */
         } 
         fmt::println("TABLE"); 
         { 
           // Print table 
           /*for (int64_t i = aliasTable.size() - 1; i >= 0; --i) { */
           /*  fmt::println("{} = ({},{})", i, aliasTable[i].p, aliasTable[i].a); */
           /*} */
         } 
        // 8. Test pack invariants
        {

            SPDLOG_DEBUG("Testing results");
            const auto err =
                wrs::test::pmr::assert_is_alias_table<weight_t, weight_t, wrs::glsl::uint>(
                    weights, aliasTable, totalWeight, 0.01, resource);
            if (err) {
                SPDLOG_ERROR(err.message());
            }
        }

        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::simd_pack::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing simd pack algorithm");
    const TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{buffers.aliasTable->get_size() * 10};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        renderdoc::startCapture();
        switch (testCase.weightType) {
        case WEIGHT_TYPE_FLOAT:
            bool failed = runTestCase<float>(testContext, testCase, buffers, stage, resource);
            if (failed) {
                failCount += 1;
            }
            break;
        }
        renderdoc::stopCapture();
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
