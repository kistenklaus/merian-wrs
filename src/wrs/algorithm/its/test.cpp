#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/renderdoc.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/test/test.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <tuple>

#include "./ITS.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = ITS;
using Buffers = Algorithm::Buffers;

struct TestCase {
    glsl::uint N;
    Distribution distribution;
    glsl::uint S;

    glsl::uint prefixSumWorkgroupSize;
    glsl::uint prefixSumRows;
    glsl::uint prefixSumLookbackDepth;
    glsl::uint samplingWorkgroupSize;
    glsl::uint cooperativeSamplingSize;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .N = 1024 * 2048,
        .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = 1024 * 2048 / 32,
        .prefixSumWorkgroupSize = 512,
        .prefixSumRows = 8,
        .prefixSumLookbackDepth = 32,
        .samplingWorkgroupSize = 512,
        .cooperativeSamplingSize = 4096,
        .iterations = 1,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    glsl::uint maxWeightCount;
    glsl::uint maxSamplesCount;
    glsl::uint maxPartitionSize;
    for (const auto& testCase : TEST_CASES) {
        maxWeightCount = std::max(maxWeightCount, testCase.N);
        maxSamplesCount = std::max(maxSamplesCount, testCase.S);
        maxPartitionSize =
            std::max(maxPartitionSize, testCase.prefixSumWorkgroupSize * testCase.prefixSumRows);
    }

    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxWeightCount, maxSamplesCount, maxPartitionSize);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                      maxWeightCount, maxSamplesCount, maxPartitionSize);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const vk::CommandBuffer cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> weights) {
    Buffers::WeightsView stageView{stage.weights, weights.size()};
    Buffers::WeightsView localView{buffers.weights, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void
downloadToStage(vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage, std::size_t S) {
    Buffers::SamplesView stageView{stage.samples, S};
    Buffers::SamplesView localView{buffers.samples, S};
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

struct Results {
    std::pmr::vector<glsl::uint> samples;
};
static Results
downloadFromStage(Buffers& stage, std::size_t S, std::pmr::memory_resource* resource) {
    Buffers::SamplesView stageView{stage.samples, S};
    auto samples = stageView.download<glsl::uint, wrs::pmr_alloc<glsl::uint>>(resource);
    return Results{
        .samples = std::move(samples),
    };
};

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName = fmt::format(
        "{{N={},S={},Dist={},PWG={},PR={},PD={},SWG={},CSS={}}}", testCase.N, testCase.S,
        distribution_to_pretty_string(testCase.distribution), testCase.prefixSumWorkgroupSize,
        testCase.prefixSumRows, testCase.prefixSumLookbackDepth, testCase.samplingWorkgroupSize,
        testCase.cooperativeSamplingSize);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context,
                     testCase.prefixSumWorkgroupSize,
                     testCase.prefixSumRows,
                     testCase.prefixSumLookbackDepth,
                     testCase.samplingWorkgroupSize,
                     testCase.cooperativeSamplingSize};

    bool failed = false;
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
        auto weights = wrs::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
        // TODO
        context.profiler->end();

        // 2. Begin recoding
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload test case indices
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload test case");
            SPDLOG_DEBUG("Uploading test case...");
            uploadTestCase(cmd, buffers, stage, weights);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N, testCase.S, context.profiler);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.S);
        }

        // 6. Submit to device
        context.profiler->end();
        context.profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd.end();
        context.queue->submit_wait(cmd);

        // 7. Download from stage
        context.profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results results = downloadFromStage(stage, testCase.S, resource);
        context.profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");
            // TODO
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::its::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing full inverse transform sampling algorithm");

    const TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{4096 * 2048};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        renderdoc::startCapture();
        runTestCase(testContext, testCase, buffers, stage, resource);
        renderdoc::stopCapture();
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
