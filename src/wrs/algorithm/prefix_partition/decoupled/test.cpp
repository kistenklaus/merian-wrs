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

#include "./DecoupledPrefixPartition.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = DecoupledPrefixPartition;
using Buffers = Algorithm::Buffers;

struct TestCase {
    DecoupledPrefixPartitionConfig config;
    glsl::uint N;
    Distribution dist;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = {},
        .N = 1024 * 2048,
        .dist = wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 1,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    glsl::uint maxPartitionSize = 0;
    glsl::uint maxN = 0;
    for (const auto& testCase : TEST_CASES) {
        maxPartitionSize = std::max(maxPartitionSize, testCase.config.partitionSize());
        maxN = std::max(maxN, testCase.N);
    }

    Buffers stage = Buffers::allocate(context.alloc, maxN, maxPartitionSize,
                                      merian::MemoryMappingType::HOST_ACCESS_RANDOM);
    Buffers local =
        Buffers::allocate(context.alloc, maxN, maxPartitionSize, merian::MemoryMappingType::NONE);

    return {local, stage};
}

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> elements) {
    {
        Buffers::ElementsView stageView{stage.elements, elements.size()};
        Buffers::ElementsView localView{buffers.elements, elements.size()};
        stageView.upload(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
}

static void downloadToStage([[maybe_unused]] const merian::CommandBufferHandle& cmd,
                            [[maybe_unused]] Buffers& buffers,
                            [[maybe_unused]] Buffers& stage) {}

struct Results2 {
    // TODO
};
static Results2 downloadFromStage([[maybe_unused]] Buffers& stage,
                                 [[maybe_unused]] std::pmr::memory_resource* resource) {
    return Results2{};
};

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{workgroupSize={},N={}}}", testCase.config.workgroupSize, testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

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
        const auto& weights = wrs::generate_weights(testCase.dist, testCase.N);
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

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage);
        }

        // 6. Submit to device
        context.profiler->end();
        context.profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd->end();
        context.queue->submit_wait(cmd);

        // 7. Download from stage
        context.profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");

        [[maybe_unused]] Results2 results = downloadFromStage(stage, resource);

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

void wrs::test::decoupled_prefix_partition::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing DecoupledPrefixPartition algorithm");

    const TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{4096 * 2048};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, buffers, stage, resource);
        renderdoc::stopCapture();
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
