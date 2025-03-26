#include "./test.hpp"
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 19/02/2025
 * @filename    : test.cpp
 */

#include "merian/vk/utils/profiler.hpp"
#include "src/device/mean/Mean.hpp"
#include "src/device/mean/decoupled/DecoupledMean.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/reference/mean.hpp"
#include "src/host/test/context.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <format>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>

// NOTE: Bad quick hack
#ifndef MERIAN_PROFILER_ENABLE
#define MERIAN_PROFILER_ENABLE
#endif

namespace device::mean {

using namespace device;
using namespace host;
using namespace host::test;

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
        .config = DecoupledMeanConfig(),
        .N = static_cast<uint32_t>(1e4),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledMeanConfig(),
        .N = static_cast<uint32_t>(1e5),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledMeanConfig(),
        .N = static_cast<uint32_t>(1e6),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = AtomicMeanConfig(),
        .N = static_cast<uint32_t>(1e4),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = AtomicMeanConfig(),
        .N = static_cast<uint32_t>(1e5),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = AtomicMeanConfig(),
        .N = static_cast<uint32_t>(1e6),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = AtomicMeanConfig(),
        .N = static_cast<uint32_t>(1e7),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = AtomicMeanConfig(),
        .N = static_cast<uint32_t>(1e8),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .iterations = 2,
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

void runTestCase(const TestContext& context,
                 const TestCase& testCase,
                 std::pmr::memory_resource* resource,
                 const merian::ProfilerHandle& profiler) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(context.queue);

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

    std::string label = fmt::format("{{{}-{}}}", meanConfigName(testCase.config), testCase.N);

    std::string recordingLabel = fmt::format("Recoding: {}", label);

    host::test::TestResultType out = host::test::TestResultType::SUCCESS;
    for (size_t i = 0; i < testCase.iterations; ++i) {
        context.queue->wait_idle();

        if (testCase.iterations > 1) {
            if (testCase.N > 5e5) {
                SPDLOG_INFO(
                    fmt::format("Testing iteration {} out of {}", i + 1, testCase.iterations));
            }
        }

        profiler->start(label);

        // Generate elements
        std::pmr::vector<base> elements{resource};
        {
            SPDLOG_DEBUG(fmt::format("Generating {} elements with {}", testCase.N,
                                     host::distribution_to_pretty_string(testCase.distribution)));
            profiler->start("Generate elements");
            elements =
                host::pmr::generate_weights<base>(testCase.distribution, testCase.N, resource);
            profiler->end();
        }

        // Begin recording
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();

        profiler->start(recordingLabel);
        profiler->cmd_start(cmd, recordingLabel);

        // Upload elements
        {
            SPDLOG_DEBUG("Uploading elements");

            profiler->start("Uploading elements");
            profiler->cmd_start(cmd, "Uploading elements");
            uploadTestCase(cmd, elements, buffers, stage);
            profiler->end();
            profiler->cmd_end(cmd);
        }

        // Run algorithm
        {
            SPDLOG_DEBUG("Running algorithm");

            profiler->start("Execute Algorithm");
            profiler->cmd_start(cmd, "Execute Algorithm");
            kernel.run(cmd, buffers, testCase.N);
            profiler->end();
            profiler->cmd_end(cmd);
        }

        // Download results to stage
        {
            profiler->start("Download result to stage");
            profiler->cmd_start(cmd, "Download result to stage");
            downloadToStage(cmd, buffers, stage);
            profiler->end();
            profiler->cmd_end(cmd);
        }

        // Submit to queue
        {
            profiler->end();
            profiler->cmd_end(cmd);
            cmd->end();
            context.queue->submit_wait(cmd);
        }

        // Download results from stage
        base mean;
        {
            mean = downloadFromStage(stage);
        }

        // Compute reference
        base referenceMean = host::reference::mean<base, host::pmr_alloc<base>>(elements, resource);

        if (std::abs(referenceMean - mean) > 0.1) {
            SPDLOG_ERROR(fmt::format("{} is just wrong\n"
                                     "Expected {}, Got{}",
                                     label, referenceMean, mean));
            out += host::test::ERROR;
        } else if (std::abs(referenceMean - mean) > 0.01) {
            SPDLOG_ERROR(fmt::format("{} is numerically unstable\n"
                                     "Expected {}, Got{}",
                                     label, referenceMean, mean));
            out += host::test::WARNING;
        }

        profiler->end();

        profiler->collect(true, true);
    }
    const auto report = profiler->get_report().gpu_report;
    auto recordingEntry =
        std::ranges::find_if(report, [&](const merian::Profiler::ReportEntry& entry) {
            return entry.name == recordingLabel;
        });
    if (recordingEntry == report.end()) {
        throw std::runtime_error("Impossible state");
    }
    auto entry = std::ranges::find_if(recordingEntry->children,
                                      [&](const merian::Profiler::ReportEntry& entry) {
                                          return entry.name == "Execute Algorithm";
                                      });
    if (entry == recordingEntry->children.end()) {
        throw std::runtime_error("Impossible state");
    }

    context.pushResult(
        "Mean", meanConfigClass(testCase.config), out, entry->duration,
        entry->std_deviation,
        {TestProperty{.name = "N",
                      .value = std::format("{:.0e}", static_cast<float>(testCase.N))}});
}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing Mean algorithm");

    SPDLOG_DEBUG("Allocating buffers");

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context.context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context.context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, testCase, context.memory_resource, profiler);
        context.resetMemoryResource();
    }

    profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results (Mean): \n{}",
                            merian::Profiler::get_report_str(profiler->get_report())));
}

} // namespace device::mean
