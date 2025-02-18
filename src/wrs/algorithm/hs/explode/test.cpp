#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : Explode.hpp
 */

#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/eval/histogram.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/sample_alias_table.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/reference/sweeping_alias_table.hpp"
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
#include <vulkan/vulkan_structs.hpp>

#include "./Explode.hpp"
#include "src/wrs/why.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = Explode;
using Buffers = Algorithm::Buffers;

struct TestCase {
    glsl::uint workgroupSize;
    glsl::uint rows;
    glsl::uint lookbackDepth;
    glsl::uint N;
    Distribution distribution;
    glsl::uint S;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    /* TestCase{ */
    /*     .workgroupSize = 512, */
    /*     .rows = 8, */
    /*     .lookbackDepth = 32, */
    /*     .N = 1024 * 2048, */
    /*     .distribution = wrs::Distribution::PSEUDO_RANDOM_UNIFORM, */
    /*     .S = 1024 * 2048 / 64, */
    /*     .iterations = 1, */
    /* }, */

    TestCase{
        .workgroupSize = 64,
        .rows = 1,
        .lookbackDepth = 32,
        .N = 64,
        .distribution = wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 64,
        .iterations = 1,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {

    glsl::uint maxN = 0;
    glsl::uint maxS = 0;
    glsl::uint maxPartitionSize = 0;
    for (const auto& testCase : TEST_CASES) {
        maxN = std::max(maxN, testCase.N);
        maxS = std::max(maxS, testCase.S);
        maxPartitionSize = std::max(maxPartitionSize, testCase.workgroupSize * testCase.rows);
    }

    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxN, maxS, maxPartitionSize);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, maxN, maxS,
                                      maxPartitionSize);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const glsl::uint> outputSensitiveSamples,
                           std::size_t partitionSize) {
    {
        Buffers::OutputSensitiveView stageView{stage.outputSensitive,
                                               outputSensitiveSamples.size()};
        Buffers::OutputSensitiveView localView{buffers.outputSensitive,
                                               outputSensitiveSamples.size()};
        stageView.upload(outputSensitiveSamples);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        std::size_t N = outputSensitiveSamples.size();
        std::size_t workgroupCount = (N + partitionSize - 1) / partitionSize;
        Buffers::DecoupledStatesView localView{buffers.decoupledState, workgroupCount};
        localView.zero(cmd);
        localView.expectComputeRead(cmd);
    }
}

static void downloadToStage(const merian::CommandBufferHandle& cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            std::size_t S) {
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
    std::string testName = fmt::format("{{workgroupSize={},N={},S={}}}", testCase.workgroupSize,
                                       testCase.N, testCase.S);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.workgroupSize, testCase.rows,
                     testCase.lookbackDepth};

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
        SPDLOG_DEBUG("Generating input");
        std::pmr::vector<float> weights =
            wrs::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
        float totalWeight = wrs::reference::kahan_reduction<float>(weights);
        wrs::pmr::AliasTable<float, glsl::uint> aliasTable =
            wrs::reference::pmr::sweeping_alias_table<float, float, glsl::uint>(
                weights, totalWeight, resource);

        auto referenceSamples = wrs::reference::pmr::sample_alias_table<float, glsl::uint>(
            aliasTable, testCase.S, resource);
        std::pmr::vector<glsl::uint> outputSensitive =
            wrs::eval::histogram<glsl::uint, wrs::pmr_alloc<glsl::uint>>(referenceSamples,
                                                                         testCase.N, resource);

        if (testCase.N < 1024) {
            fmt::println("Histogram");
            for (std::size_t i = 0; i < testCase.N; ++i) {
                fmt::println("[{}]: {}", i, outputSensitive[i]);
            }
        }

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
            uploadTestCase(cmd, buffers, stage, outputSensitive,
                           testCase.workgroupSize * testCase.rows);
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

            if (testCase.S < 1024) {
                fmt::println("Samples");
                for (std::size_t i = 0; i < results.samples.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.samples[i]);
                }
            }
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::hs_explode::test(const merian::ContextHandle& context) {
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
