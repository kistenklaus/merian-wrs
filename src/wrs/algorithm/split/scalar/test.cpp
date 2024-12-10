#include "./test.hpp"

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/split/scalar/ScalarSplit.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_cases.hpp"
#include "src/wrs/algorithm/split/scalar/test/test_setup.hpp"
#include "src/wrs/generic_types.hpp"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/mean.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/test.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>

using namespace wrs::test::scalar_split;

vk::DeviceSize wrs::test::scalar_split::sizeOfWeightType(WeightType type) {
    switch (type) {
    case WEIGHT_TYPE_FLOAT:
        return sizeof(float);
    }
    throw std::runtime_error("sizeOfWeightType is not implemented properly");
}

template <typename weight_t>
static void uploadTestCase(vk::CommandBuffer cmd,
                           const std::span<weight_t>& heavyPrefix,
                           const std::span<weight_t>& reverseLightPrefix,
                           const weight_t mean,
                           Buffers& buffers,
                           Buffers& stage) {
    { // Upload heavy & light prefix sums
        void* partitionPrefixMapped = stage.partitionPrefix->get_memory()->map();
        uint32_t* heavyCountMapped = reinterpret_cast<uint32_t*>(partitionPrefixMapped);
        float* heavyPrefixMapped = reinterpret_cast<float*>(heavyCountMapped + 1);
        float* lightPrefixMapped = reinterpret_cast<float*>(heavyPrefixMapped + heavyPrefix.size());

        *heavyCountMapped = static_cast<uint32_t>(heavyPrefix.size());
        std::memcpy(heavyPrefixMapped, heavyPrefix.data(), heavyPrefix.size() * sizeof(weight_t));
        std::memcpy(lightPrefixMapped, reverseLightPrefix.data(),
                    reverseLightPrefix.size() * sizeof(weight_t));

        stage.partitionPrefix->get_memory()->unmap();

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                            {}, {},
                            stage.partitionPrefix->buffer_barrier(
                                vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eTransferRead),
                            {});

        size_t N = heavyPrefix.size() + reverseLightPrefix.size();
        vk::BufferCopy copy{0, 0, Buffers::minPartitionPrefixBufferSize(N, sizeof(weight_t))};
        cmd.copyBuffer(*stage.partitionPrefix, *buffers.partitionPrefix, 1, &copy);

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {},
            buffers.partitionPrefix->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                    vk::AccessFlagBits::eShaderRead),
            {});
    }
    { // Upload mean
        weight_t* meanMapped = stage.mean->get_memory()->map_as<weight_t>();
        *meanMapped = mean;
        stage.mean->get_memory()->unmap();
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                            {}, {},
                            stage.mean->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                       vk::AccessFlagBits::eTransferRead),
                            {});

        vk::BufferCopy copy{0, 0, Buffers::minMeanBufferSize(sizeof(weight_t))};
        cmd.copyBuffer(*stage.mean, *buffers.mean, 1, &copy);

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                            buffers.mean->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                         vk::AccessFlagBits::eShaderRead),
                            {});
    }
}

template <typename weight_t>
void downloadResultsToStage(vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage, uint32_t K) {

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer, {}, {},
                        buffers.splits->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                       vk::AccessFlagBits::eTransferRead),
                        {});

    vk::BufferCopy copy{0, 0, Buffers::minSplitBufferSize(K, sizeof(weight_t))};
    cmd.copyBuffer(*buffers.splits, *stage.splits, 1, &copy);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
                        {},
                        stage.splits->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                     vk::AccessFlagBits::eHostRead),
                        {});
}

template <typename weight_t>
std::pmr::vector<Buffers::Split<weight_t>>
downloadResultsFromStage(Buffers& stage, uint32_t K, std::pmr::memory_resource* resource) {
    using Split = Buffers::Split<weight_t>;
    std::pmr::vector<Split> splits(K, resource);

    void* splitsMapped = stage.splits->get_memory()->map();
    std::memcpy(splits.data(), splitsMapped, K * sizeof(Split));
    stage.splits->get_memory()->unmap();
    return std::move(splits);
}

template <typename weight_t>
static void runTestCase(const wrs::test::TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    using Split = Buffers::Split<weight_t>;

    SPDLOG_INFO(fmt::format("Running test case for ScalarSplit:\n"
                            "\t-weightCount = {}\n"
                            "\t-splitCount = {}\n"
                            "\t-distribution = {}\n"
                            "\t-iterations = {}\n", //
                            testCase.weightCount, testCase.splitCount,
                            wrs::distribution_to_pretty_string(testCase.distribution),
                            testCase.iterations));

    SPDLOG_DEBUG("Creating ScalarSplit instance");
    wrs::ScalarSplit<weight_t> algo{context.context};

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
            fmt::format("TestCase: [weightCount={},splitCount={},distribution={}]",
                        testCase.weightCount, testCase.splitCount,
                        wrs::distribution_to_pretty_string(testCase.distribution)));

        // ===== Generate input data ======

        context.profiler->start("Generate input data");

        // 1. Generate weights
        std::pmr::vector<weight_t> weights{resource};
        {
            SPDLOG_DEBUG(fmt::format("Generating {} weights with {}", testCase.weightCount,
                                     wrs::distribution_to_pretty_string(testCase.distribution)));
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            weights = std::move(wrs::pmr::generate_weights<weight_t>(
                testCase.distribution, testCase.weightCount, resource));
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
        std::pmr::vector<weight_t> _hack{resource};
        auto partitionStorage =
            std::make_tuple<std::span<weight_t>, std::span<weight_t>, std::pmr::vector<weight_t>>(
                {}, {}, std::move(_hack));
        {
            SPDLOG_DEBUG("Compute partition");
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute partitions");
            auto temp =
                wrs::reference::pmr::stable_partition<weight_t>(weights, averageWeight, resource);
            // Swaps memory without reallocation or copy
            std::swap(temp, partitionStorage);
        }
        const auto& [heavyPartition, lightPartition, _] = partitionStorage;

        // 4. Compute prefix sums
        std::pmr::vector<weight_t> heavyPrefixSum{resource};
        std::pmr::vector<weight_t> reverseLightPrefixSum{resource};
        {
            SPDLOG_DEBUG("Compute partition prefix sums");
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute partition prefix sums");
            heavyPrefixSum =
                wrs::reference::pmr::prefix_sum<weight_t>(heavyPartition, false, resource);
            reverseLightPrefixSum =
                wrs::reference::pmr::prefix_sum<weight_t>(lightPartition, false, resource);
            std::reverse(reverseLightPrefixSum.begin(),
                         reverseLightPrefixSum.end()); // required by the layout!
        }
        uint32_t N = static_cast<uint32_t>(heavyPrefixSum.size() + reverseLightPrefixSum.size());
        uint32_t K = testCase.splitCount;
        context.profiler->end();

        // ============== Compute reference ===============
        std::pmr::vector<wrs::split_t<weight_t, uint32_t>> reference{resource};
        {
            SPDLOG_DEBUG("Compute reference split (CPU)");
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute reference split (CPU)");
            std::pmr::vector<weight_t> lightPrefix = reverseLightPrefixSum;
            std::reverse(lightPrefix.begin(), lightPrefix.end());
            reference = std::move(wrs::reference::pmr::splitK<weight_t, uint32_t>(
                heavyPrefixSum, lightPrefix, averageWeight, N, K, resource));
        }

        // =========== Start Recoding =========
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();

        std::string recordingLabel =
            fmt::format("Recoding: [weightCount={},splitCount= {}"
                        "distribution={},it={}]",
                        testCase.weightCount, testCase.splitCount,
                        wrs::distribution_to_pretty_string(testCase.distribution), i + 1);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // ================ Upload Input ===============
        {
            SPDLOG_DEBUG("Uploading input");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload partition prefix & average");
            uploadTestCase<weight_t>(cmd, heavyPrefixSum, reverseLightPrefixSum, averageWeight,
                                     buffers, stage);
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
            downloadResultsToStage<weight_t>(cmd, buffers, stage, K);
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
        std::pmr::vector<Split> splits{resource};
        {
            SPDLOG_DEBUG("Download results from stage");
            MERIAN_PROFILE_SCOPE(context.profiler, "Download results from stage");
            splits = downloadResultsFromStage<weight_t>(stage, K, resource);
        }

        // ========= Compare results against reference ==========
        {
            SPDLOG_DEBUG("Comparing results against reference");
            MERIAN_PROFILE_SCOPE(context.profiler, "Compare results against reference");
            constexpr size_t MAX_LOG = 10;
            size_t errorCounter = 0;
            for (size_t i = 0; i < K; ++i) {
                const auto& split = splits.at(i);
                const auto& ref = reference.at(i);
                bool matches = true;
                if (std::get<0>(ref) != split.i) {
                    matches = false;
                }
                if (std::get<1>(ref) != split.j) {
                    matches = false;
                }
                if (std::abs(std::get<2>(ref) - split.spill) > 0.01) {
                    matches = false;
                }
                if (!matches) {
                    if (errorCounter < MAX_LOG) {
                        SPDLOG_ERROR(fmt::format("ScalarSplit does not match the reference\n"
                                                 "Expected ({}, {}, {}), Got ({}, {}, {})",
                                                 std::get<0>(ref), std::get<1>(ref),
                                                 std::get<2>(ref), split.i, split.j, split.spill));
                    }
                    errorCounter += 1;
                }
            }
            if (errorCounter != 0) {
                SPDLOG_ERROR(fmt::format("ScalarSplit does not match the reference\n"
                                         "Assention failed at {} out of {} indicies",
                                         errorCounter, K));
            }
        }
    }
}

void wrs::test::scalar_split::test(const merian::ContextHandle& context) {

    wrs::test::TestContext testContext = wrs::test::setupTestContext(context);

    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{buffers.partitionPrefix->get_size() * 10};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    for (const auto& testCase : TEST_CASES) {
        switch (testCase.weightType) {
        case WEIGHT_TYPE_FLOAT:
            runTestCase<float>(testContext, testCase, buffers, stage, resource);
            break;
        default:
            throw std::runtime_error("FATAL");
        }
    }

    testContext.profiler->collect();
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));
}
