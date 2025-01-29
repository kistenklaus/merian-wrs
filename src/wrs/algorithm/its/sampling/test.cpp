#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/renderdoc.hpp"
#include "src/wrs/algorithm/its/sampling/InverseTransformSampling.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/test/test.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vector>

using namespace wrs;
using namespace wrs::test;

using Algorithm = InverseTransformSampling;
using Buffers = Algorithm::Buffers;

struct TestCase {
    InverseTransformSamplingConfig config;
    glsl::uint weightCount;
    wrs::Distribution distribution;
    glsl::uint sampleCount;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = {},
        .weightCount = static_cast<glsl::uint>(1e6),
        .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
        .sampleCount = static_cast<glsl::uint>(1e6),
        .iterations = 16,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    glsl::uint maxSampleCount = 0;
    glsl::uint maxWeightCount = 0;
    for (auto testCase : TEST_CASES) {
        maxSampleCount = std::max(maxSampleCount, testCase.sampleCount);
        maxWeightCount = std::max(maxWeightCount, testCase.weightCount);
    }

    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxWeightCount, maxSampleCount);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                      maxWeightCount, maxSampleCount);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const vk::CommandBuffer cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> cmf) {
    Buffers::CMFView stageView{stage.cmf, cmf.size()};
    Buffers::CMFView localView{buffers.cmf, cmf.size()};
    stageView.upload<float>(cmf);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void
downloadToStage(vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage, std::size_t sampleCount) {
    Buffers::SamplesView stageView{stage.samples, sampleCount};
    Buffers::SamplesView localView{buffers.samples, sampleCount};
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}
// TODO

struct Results {
    std::pmr::vector<glsl::uint> samples;
};

static Results
downloadFromStage(Buffers& stage, std::pmr::memory_resource* resource, std::size_t sampleCount) {
    Buffers::SamplesView stageView{stage.samples, sampleCount};
    std::pmr::vector<glsl::uint> samples =
        stageView.download<glsl::uint, std::pmr::polymorphic_allocator<glsl::uint>>(resource);
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
        fmt::format("{{workgroupSize={},weightCount={},distribution={},sampleCount={}}}",
                    testCase.config.workgroupSize, testCase.weightCount,
                    distribution_to_pretty_string(testCase.distribution), testCase.sampleCount);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, testCase.config};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(context.profiler, testName);
        context.queue->wait_idle();
        if (testCase.iterations > 1) {
            if (testCase.sampleCount > 1e6) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            }
        }

        // 1. Generate input
        context.profiler->start("Generate test input");
        std::pmr::vector<float> weights = wrs::pmr::generate_weights<float>(
            testCase.distribution, testCase.weightCount, resource);
        std::pmr::vector<float> cmf = wrs::reference::pmr::prefix_sum<float>(weights, resource);
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
            uploadTestCase(cmd, buffers, stage, cmf);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.weightCount, testCase.sampleCount);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.sampleCount);
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
        Results results = downloadFromStage(stage, resource, testCase.sampleCount);
        context.profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");

            for (glsl::uint s : results.samples) {
                // fmt::println("Sample: {}", s);
            }
            // TODO
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::its_sampling::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing Inverse Transform Sampling (ITS) algorithm");

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
