#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/renderdoc.hpp"
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
#include <tuple>

#include "./PSA.hpp"
#include "src/wrs/why.hpp"

namespace wrs::test::psa {

using namespace wrs;
using namespace wrs::test;

using Algorithm = PSA;
using Buffers = Algorithm::Buffers;

struct TestCase {
    glsl::uint N;
    Distribution distribution;
    glsl::uint S;

    PSAConfig config;

    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .N = 1024 * 2048,
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .S = 1024 * 2048 / 32,
        .config = PSAConfig::defaultV(),
        .iterations = 1,
    },
    TestCase{
        .N = static_cast<glsl::uint>(1e7),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<glsl::uint>(1e7) / 32,
        .config = PSAConfig::defaultV(),
        .iterations = 100,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {

    glsl::uint maxWeightCount = 0;
    glsl::uint maxPrefixPartitionSize = 0;
    glsl::uint maxSplitCount = 0;
    glsl::uint maxSampleCount = 0;

    for (const auto& testCase : TEST_CASES) {
        maxWeightCount = std::max(maxWeightCount, testCase.N);
        maxPrefixPartitionSize =
            std::max(maxPrefixPartitionSize, testCase.config.psac.prefixPartitionConfig.partitionSize());
        std::size_t splitSize = testCase.config.psac.splitSize;
        glsl::uint splitCount = (testCase.N + splitSize - 1) / splitSize;
        maxSplitCount = std::max(maxSplitCount, splitCount);
        maxSampleCount = std::max(maxSampleCount, testCase.S);
    }

    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxWeightCount, maxPrefixPartitionSize,
                                      maxSplitCount, maxSampleCount);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                      maxWeightCount, maxPrefixPartitionSize,
                                      maxSplitCount, maxSampleCount);

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
    std::string testName =
        fmt::format("{{N={},Dist={},S={}}}", testCase.N,
                    distribution_to_pretty_string(testCase.distribution), testCase.S);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context};

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
        const auto weights =
            wrs::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
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

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing TODO algorithm");

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

} // namespace wrs::test::psa
