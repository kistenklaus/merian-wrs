
#pragma once
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : test.cpp
 */

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
#include <fmt/base.h>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tuple>

#include "./HSTC.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = HSTC;
using Buffers = Algorithm::Buffers;

struct TestCase {
    glsl::uint workgroupSize;
    glsl::uint N;
    Distribution distribution;
    glsl::uint svoThreshold;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .workgroupSize = 512,
        .N = 16,
        .distribution = wrs::Distribution::UNIFORM,
        .svoThreshold = 512,
        .iterations = 1,
    },
    /* TestCase{ */
    /*     .workgroupSize = 32, */
    /*     .N = 3, */
    /*     .distribution = wrs::Distribution::UNIFORM, */
    /*     .iterations = 1, */
    /* }, */
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {

    glsl::uint maxN = 0;
    for (const auto& testCase : TEST_CASES) {
        maxN = std::max(maxN, testCase.N);
    }

    Buffers stage =
        Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, maxN);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, maxN);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> weights) {
    Buffers::TreeView stageView{stage.tree, weights.size()};
    Buffers::TreeView localView{buffers.tree, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void downloadToStage(const merian::CommandBufferHandle& cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            std::size_t N) {
    Buffers::TreeView stageView{stage.tree, 2 * N - 2};
    Buffers::TreeView localView{buffers.tree, 2 * N - 2};
    localView.expectComputeRead(cmd);
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

struct Results {
    std::pmr::vector<float> tree;
};
static Results
downloadFromStage(Buffers& stage, std::size_t N, std::pmr::memory_resource* resource) {
    Buffers::TreeView stageView{stage.tree, 2 * N - 2};
    auto tree = stageView.download<float, wrs::pmr_alloc<float>>(resource);
    return Results{
        .tree = std::move(tree),
    };
};

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{workgroupSize={},N={}}}", testCase.workgroupSize, testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.workgroupSize};

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
            wrs::pmr::generate_weights(testCase.distribution, testCase.N);
        context.profiler->end();

        // 2. Begin recoding
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        const std::string recordingLabel = fmt::format("Recording : {}", testName);
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
            kernel.run(cmd, buffers, testCase.N, testCase.svoThreshold);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N);
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
        Results results = downloadFromStage(stage, testCase.N, resource);
        context.profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");
            /* SPDLOG_WARN(fmt::format("tree-size: {}", results.tree.size())); */

            if (testCase.N < 1024) {
                for (std::size_t i = 0; i < results.tree.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.tree[i]);
                }
            }

            fmt::println("Root: {}", results.tree.back() + results.tree[results.tree.size() - 2]);
            fmt::println("Reduction: {}", wrs::reference::kahan_reduction<float>(weights));
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::hstc::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing HSTC algorithm");

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
