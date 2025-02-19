/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : test.cpp
 */

#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
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
#include <fmt/base.h>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tuple>

#include "./HS.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = HS;
using Buffers = Algorithm::Buffers;

struct TestCase {
    glsl::uint hstcWorkgroupSize;
    glsl::uint svoWorkgroupSize;
    glsl::uint samplingWorkgroupSize;
    glsl::uint explodeWorkgroupSize;
    glsl::uint explodeRows;
    glsl::uint explodeLookbackDepth;
    glsl::uint N;
    Distribution distribution;
    glsl::uint S;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .hstcWorkgroupSize = 512,
        .svoWorkgroupSize = 512,
        .samplingWorkgroupSize = 512,
        .explodeWorkgroupSize = 512,
        .explodeRows = 8,
        .explodeLookbackDepth = 32,
        .N = 1024 * 2048,
        .distribution = Distribution::UNIFORM,
        .S = 1024 * 2048 / 64,
        .iterations = 1,
    },
    /* TestCase{  */
    /*     .hstcWorkgroupSize = 512,  */
    /*     .svoWorkgroupSize = 512, */
    /*     .samplingWorkgroupSize = 512,  */
    /*     .explodeWorkgroupSize = 32,  */
    /*     .explodeRows = 1,  */
    /*     .explodeLookbackDepth = 1,  */
    /*     .N = 1024,  */
    /*     .distribution = Distribution::UNIFORM,  */
    /*     .S = 64,  */
    /*     .iterations = 1,  */
    /* },  */
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    glsl::uint maxN = 0;
    glsl::uint maxS = 0;
    glsl::uint maxExplodePartitionSize = 0;

    for (const auto& testCase : TEST_CASES) {
        maxN = std::max(maxN, testCase.N);
        maxS = std::max(maxS, testCase.S);
        maxExplodePartitionSize =
            std::max(maxExplodePartitionSize, testCase.explodeWorkgroupSize * testCase.explodeRows);
    }

    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxN, maxS, maxExplodePartitionSize);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, maxN, maxS,
                                      maxExplodePartitionSize);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const merian::CommandBufferHandle cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> weights,
                           std::size_t S) {

    // Upload weights
    Buffers::WeightTreeView stageView{stage.weightTree, weights.size()};
    Buffers::WeightTreeView localView{buffers.weightTree, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);

    {
        // Upload sample count
        wrs::hst::HSTRepr repr{weights.size()};
        glsl::uint* mapped = stage.outputSensitiveSamples->get_memory()->map_as<glsl::uint>();
        mapped[repr.size()] = S;
        stage.samples->get_memory()->unmap();
        Buffers::OutputSensitiveSamplesView stageView{stage.outputSensitiveSamples,
                                                      repr.size() + 1};
        Buffers::OutputSensitiveSamplesView localView{buffers.outputSensitiveSamples,
                                                      repr.size() + 1};
        stageView.expectHostWrite();
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
}

static void
downloadToStage(merian::CommandBufferHandle cmd, Buffers& buffers, Buffers& stage, std::size_t S) {
    Buffers::SamplesView stageView{stage.samples, S};
    Buffers::SamplesView localView{buffers.samples, S};
    localView.expectComputeWrite();
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
    std::string testName = fmt::format("{{N={},S={}}}", testCase.N, testCase.S);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{
        context.context,           context.shaderCompiler,         testCase.hstcWorkgroupSize,
        testCase.svoWorkgroupSize, testCase.samplingWorkgroupSize, testCase.explodeWorkgroupSize,
        testCase.explodeRows,      testCase.explodeLookbackDepth};

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
        std::pmr::vector<float> weights =
            wrs::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
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
            uploadTestCase(cmd, buffers, stage, weights, testCase.S);
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
        cmd->end();
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
            if (testCase.S <= 1024) {
                for (std::size_t i = 0; i < results.samples.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.samples[i]);
                }
            }
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::hs::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing Hierarchical Sampling algorithm");

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
