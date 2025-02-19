#include "./test.hpp"
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 19/02/2025
 * @filename    : test.cpp
 */

#include "merian/vk/utils/profiler.hpp"
#include "src/device/mean/Mean.hpp"
#include "src/device/mean/decoupled/DecoupledMean.hpp"
#include "src/host/assert/test.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/memory/FallbackResource.hpp"
#include "src/host/memory/SafeResource.hpp"
#include "src/host/memory/StackResource.hpp"
#include "src/host/reference/mean.hpp"
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>

using namespace device;
using namespace host;
using namespace host::test;

namespace device::test::mean {

using base = host::glsl::f32;
using Algorithm = device::Mean<base>;
using Buffers = Algorithm::Buffers;
using Config = Algorithm::Config;

struct TestCase {
    Config config;
    uint32_t N;
    Distribution distribution;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = AtomicMeanConfig(),
        .N = static_cast<uint32_t>(1024 * 2048 + 1),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 5,
    },
    TestCase{
        .config = DecoupledMeanConfig(),
        .N = static_cast<uint32_t>(1024 * 2048 + 1),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 5,
    },
};

void uploadTestCase(const merian::CommandBufferHandle& cmd,
                    std::span<const base> elements,
                    Buffers& buffers,
                    Buffers& stage) {

    SPDLOG_DEBUG("Staged upload");
    {
        Buffers::ElementsView<base> stageView{stage.elements, elements.size()};
        Buffers::ElementsView<base> localView{buffers.elements, elements.size()};
        stageView.template upload<base>(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
}

void downloadToStage(const merian::CommandBufferHandle& cmd, Buffers& buffers, Buffers& stage) {
    Buffers::MeanView<base> stageView{stage.mean};
    Buffers::MeanView<base> localView{buffers.mean};
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

base downloadFromStage(Buffers& stage) {
    Buffers::MeanView<base> stageView{stage.mean};
    return stageView.template download<base>();
}

bool runTestCase(const TestContext& context,
                 const TestCase& testCase,
                 std::pmr::memory_resource* resource) {

    Buffers buffers = Buffers::allocate<base>(context.alloc, merian::MemoryMappingType::NONE,
                                              testCase.config, testCase.N);
    Buffers stage = Buffers::allocate<base>(
        context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, testCase.config, testCase.N);

    SPDLOG_INFO(fmt::format("Running test case for DecoupledMean:\n"
                            "\t-{}\n"
                            "\t-N = {}\n"
                            "\t-distribution = {}\n"
                            "\t-iterations = {}\n", //
                            meanConfigName(testCase.config), testCase.N,
                            host::distribution_to_pretty_string(testCase.distribution),
                            testCase.iterations));
    SPDLOG_DEBUG("Creating DecoupledMean algorithm instance");
    Algorithm kernel(context.context, context.shaderCompiler, testCase.config);

    bool failed = false;
    for (size_t i = 0; i < testCase.iterations; ++i) {
        context.queue->wait_idle();

        if (testCase.iterations > 1) {
            if (testCase.N > 5e5) {
                SPDLOG_INFO(
                    fmt::format("Testing iteration {} out of {}", i + 1, testCase.iterations));
            }
        }

        std::string label = fmt::format("{{{}-{}}}", meanConfigName(testCase.config), testCase.N);
        MERIAN_PROFILE_SCOPE(context.profiler, label);

        // Generate elements
        std::pmr::vector<base> elements{resource};
        {
            SPDLOG_DEBUG(fmt::format("Generating {} weights with {}", testCase.N,
                                     host::distribution_to_pretty_string(testCase.distribution)));
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            elements =
                host::pmr::generate_weights<base>(testCase.distribution, testCase.N, resource);
        }

        // Begin recording
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();

        std::string recordingLabel = fmt::format("Recoding: {}", label);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // Upload elements
        {
            SPDLOG_DEBUG("Uploading elements");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Uploading elements");
            uploadTestCase(cmd, elements, buffers, stage);
        }

        // Run algorithm
        {
            SPDLOG_DEBUG("Running algorithm");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute DecoupledMean");
            kernel.run(cmd, buffers, testCase.N);
        }

        // Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download result to stage");
            downloadToStage(cmd, buffers, stage);
        }

        // Submit to queue
        {
            context.profiler->end();
            context.profiler->cmd_end(cmd);
            cmd->end();

            MERIAN_PROFILE_SCOPE(context.profiler, "Wait for device idle");
            context.queue->submit_wait(cmd);
        }

        // Download results from stage
        base mean;
        {
            mean = downloadFromStage(stage);
        }

        // Compute reference
        base referenceMean = host::reference::mean<base, host::pmr_alloc<base>>(elements, resource);

        if (std::abs(referenceMean - mean) > 0.01) {
            SPDLOG_ERROR(fmt::format("{} is numerically unstable\n"
                                     "Expected {}, Got{}",
                                     label, referenceMean, mean));
            failed = true;
        }

        context.profiler->collect(true, true);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing decoupled_mean algorithm");

    TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");

    host::memory::StackResource stackResource{2048 * 4096};
    host::memory::FallbackResource fallbackResource{&stackResource};
    host::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, resource);
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results (Mean): \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));
}

} // namespace device::test::mean
