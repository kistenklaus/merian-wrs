#include "./test.hpp"

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/split/scalar/ScalarSplit.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_cases.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_setup.hpp"
#include "src/wrs/layout/layout_traits.hpp"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/mean.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_split.hpp"
#include "src/wrs/test/test.hpp"
#include "src/wrs/types/partition.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_enums.hpp>

namespace wrs::test::scalar_split {

vk::DeviceSize sizeOfWeightType(WeightType type) {
    switch (type) {
    case WEIGHT_TYPE_FLOAT:
        return sizeof(float);
    }
    throw std::runtime_error("sizeOfWeightType is not implemented properly");
}

using weight_t = float;

static void uploadTestCase(vk::CommandBuffer cmd,
                           const std::span<weight_t>& heavyPrefix,
                           const std::span<weight_t>& reverseLightPrefix,
                           const weight_t mean,
                           Buffers& buffers,
                           Buffers& stage,
                           std::pmr::memory_resource* resource) {
    { // Upload heavy & light prefix sums
        std::size_t N = heavyPrefix.size() + reverseLightPrefix.size();
        wrs::glsl::uint heavyCount = heavyPrefix.size();
        wrs::Partition<weight_t, std::pmr::vector<weight_t>> heavyLight(
            std::pmr::vector<weight_t>{N, resource}, heavyPrefix.size());

        std::memcpy(heavyLight.heavy().data(), heavyPrefix.data(), heavyPrefix.size_bytes());
        std::memcpy(heavyLight.light().data(), reverseLightPrefix.data(),
                    reverseLightPrefix.size_bytes());

        Buffers::PartitionPrefixView stageView{stage.partitionPrefix, N};
        Buffers::PartitionPrefixView localView{buffers.partitionPrefix, N};
        stageView.attribute<"heavyCount">().template upload<wrs::glsl::uint>(heavyCount);
        stageView.attribute<"heavyLightIndices">().template upload<weight_t>(heavyLight.storage());
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    { // Upload mean
         Buffers::MeanView stageView{stage.mean}; 
         Buffers::MeanView localView{buffers.mean}; 
         stageView.upload(mean); 
         stageView.copyTo(cmd, localView); 
         localView.expectComputeRead(cmd); 
    }
}

void downloadResultsToStage(vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage, uint32_t K) {
     Buffers::SplitsView stageView{stage.splits, K}; 
     Buffers::SplitsView localView{buffers.splits, K}; 
     localView.expectComputeWrite(); 
     localView.copyTo(cmd, stageView); 
     stageView.expectHostRead(cmd); 
}

std::pmr::vector<Buffers::Split<weight_t>>
downloadResultsFromStage(Buffers& stage, uint32_t K, std::pmr::memory_resource* resource);

static void runTestCase(const wrs::test::TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {

    SPDLOG_INFO(fmt::format("Running test case for ScalarSplit:\n"
                            "\t-workgroupSize={}\n"
                            "\t-weightCount = {}\n"
                            "\t-splitCount = {}\n"
                            "\t-distribution = {}\n"
                            "\t-iterations = {}\n", //
                            testCase.workgroupSize, testCase.weightCount, testCase.splitCount,
                            wrs::distribution_to_pretty_string(testCase.distribution),
                            testCase.iterations));

    SPDLOG_DEBUG("Creating ScalarSplit instance");

    wrs::ScalarSplit algo{context.context, testCase.workgroupSize};

    for (size_t i = 0; i < testCase.iterations; ++i) {

        context.queue->wait_idle();
        context.profiler->collect(true, true);

        if (testCase.iterations > 1) {
            if (testCase.weightCount > 5e5) {
                SPDLOG_INFO(
                    fmt::format("Testing iteration {} out of {}", i + 1, testCase.iterations));
            }
        }

        MERIAN_PROFILE_SCOPE(
            context.profiler,
            fmt::format("TestCase: [workgroupSize={},weightCount={},splitCount={},distribution={}]",
                        testCase.workgroupSize, testCase.weightCount, testCase.splitCount,
                        wrs::distribution_to_pretty_string(testCase.distribution)));

        // ===== Generate input data ======

        context.profiler->start("Generate input data");

        // 1. Generate weights
        std::pmr::vector<weight_t> weights{resource};
        {
            SPDLOG_DEBUG(fmt::format("Generating {} weights with {}", testCase.weightCount,
                                     wrs::distribution_to_pretty_string(testCase.distribution)));
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            weights = wrs::pmr::generate_weights<weight_t>(
                testCase.distribution, testCase.weightCount, resource);
        }
        // 2. Compute average
        weight_t averageWeight;
        {
            SPDLOG_DEBUG("Compute average weight (on CPU)");
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute average");
            averageWeight = wrs::reference::pmr::mean<weight_t>(weights, resource);
        }
        // 3. Compute partitions

        // This is required because can't to a deallocation or copy operations, because
        // it would invalidate the spans. Therefor this small hack to ensure that
        // the swap operation does not deallocate. Requires that both parterns of the swap
        // own memory from allocators that compare equal (see pmr::memory_resource
        // or the C++11 allocator_traits )
        wrs::Partition<weight_t, std::pmr::vector<weight_t>> heavyLightPartition;
        {
            SPDLOG_DEBUG("Compute partition");
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute partitions");
            heavyLightPartition =
                wrs::reference::pmr::stable_partition<weight_t>(weights, averageWeight, resource);
        }
        const auto heavyPartition = heavyLightPartition.heavy();
        const auto lightPartition = heavyLightPartition.light();

        // 4. Compute prefix sums
        std::pmr::vector<weight_t> heavyPrefixSum{resource};
        std::pmr::vector<weight_t> lightPrefixSum{resource};
        std::pmr::vector<weight_t> reverseLightPrefixSum{resource};
        {
            SPDLOG_DEBUG("Compute partition prefix sums");
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute partition prefix sums");
            heavyPrefixSum =
                wrs::reference::pmr::imperfect_prefix_sum<weight_t>(heavyPartition, 0, resource);
            lightPrefixSum =
                wrs::reference::pmr::imperfect_prefix_sum<weight_t>(lightPartition, 0, resource);
            reverseLightPrefixSum = lightPrefixSum;
            std::reverse(reverseLightPrefixSum.begin(),
                         reverseLightPrefixSum.end()); // required by the layout!
        }
        auto N = static_cast<uint32_t>(heavyPrefixSum.size() + reverseLightPrefixSum.size());
        uint32_t K = testCase.splitCount;
        context.profiler->end();

        // ============== Compute reference ===============
        /* std::pmr::vector<wrs::Split<weight_t, uint32_t>> reference{resource}; */
        /* { */
        /*     SPDLOG_DEBUG("Compute reference split (CPU)"); */
        /*     MERIAN_PROFILE_SCOPE(context.profiler, "Compute reference split (CPU)"); */
        /*     std::pmr::vector<weight_t> lightPrefix = reverseLightPrefixSum; */
        /*     std::reverse(lightPrefix.begin(), lightPrefix.end()); */
        /*     reference = std::move(wrs::reference::pmr::splitK<weight_t, uint32_t>( */
        /*         heavyPrefixSum, lightPrefix, averageWeight, N, K, resource)); */
        /* } */

        // =========== Start Recoding =========
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();

        std::string recordingLabel =
            fmt::format("Recoding: [workgroupSize={},weightCount={},splitCount= {}"
                        "distribution={},it={}]",testCase.workgroupSize,
                        testCase.weightCount, testCase.splitCount,
                        wrs::distribution_to_pretty_string(testCase.distribution), i + 1);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // ================ Upload Input ===============
        {
            SPDLOG_DEBUG("Uploading input");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload partition prefix & average");
            uploadTestCase(cmd, heavyPrefixSum, reverseLightPrefixSum, averageWeight,
                                     buffers, stage, resource);
        }

        // ============= Execute algorithm =============
        {
            SPDLOG_DEBUG("Executing algorithm");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            algo.run(cmd, buffers, N, K);
        }
        // ========= Download results to Stage =========
        {
            SPDLOG_DEBUG("Downloading results to stage");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download result to stage");
            downloadResultsToStage(cmd, buffers, stage, K);
        }

        // =============== Submit to queue ============
        {
            context.profiler->end();
            context.profiler->cmd_end(cmd);
            SPDLOG_DEBUG("Submiting to queue & waiting");
            cmd.end();
            context.queue->submit_wait(cmd);
        }

        // ========= Download results from Stage ======
        std::pmr::vector<wrs::Split<weight_t, wrs::glsl::uint>> splits{resource};
        {
            SPDLOG_DEBUG("Download results from stage");
            MERIAN_PROFILE_SCOPE(context.profiler, "Download results from stage");
            splits = downloadResultsFromStage(stage, K, resource);
        }

        // ========= Compare results against reference ==========
        {

            if (testCase.splitCount <= 1024) {
              fmt::println("");
              for (std::size_t i = 0; i < splits.size(); ++i) {
                fmt::println("({},{},{})", splits[i].i, splits[i].j, splits[i].spill);
              }
              fmt::println("");
            }

            SPDLOG_DEBUG("Testing splits");
            auto err = wrs::test::pmr::assert_is_split<weight_t, wrs::glsl::uint>(
                splits, K, heavyPrefixSum, lightPrefixSum, averageWeight, 0.01, resource);
            if (err) {
                SPDLOG_ERROR(fmt::format("Invalid split!\n{}", err.message()));
            } else {
              SPDLOG_INFO("Splits: OK");
            }
        }
        context.profiler->collect(true,true);
    }
}

void test(const merian::ContextHandle& context) {

    wrs::test::TestContext testContext = wrs::test::setupTestContext(context);

    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{buffers.partitionPrefix->get_size() * 10};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, buffers, stage, resource);
    }

    testContext.profiler->collect(true,true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));
}

}
