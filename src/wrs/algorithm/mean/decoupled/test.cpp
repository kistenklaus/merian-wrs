#include "./test.hpp"
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : test.cpp
 */

#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/mean/decoupled/DecoupledMean.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/test/test.hpp"
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>

using namespace wrs;
using namespace wrs::test;
using namespace wrs::test::decoupled_mean;

using Buffers = DecoupledMeanBuffers;

struct TestCase {
    uint32_t workgroupSize;
    uint32_t rows;
    uint32_t elementCount;
    Distribution distribution;
    bool stable;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .workgroupSize = 512,
        .rows = 8,
        .elementCount = static_cast<uint32_t>(1024 * 2048),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = false,
        .iterations = 1,
    },

    TestCase{
        .workgroupSize = 64,
        .rows = 2,
        .elementCount = 256,
        .distribution = Distribution::UNIFORM,
        .stable = false,
        .iterations = 1,
    },
};

inline std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {

    uint32_t maxElementCount = 0;
    std::size_t maxPartitionSize = 0;

    for (const auto& testCase : TEST_CASES) {
        maxElementCount = std::max(maxElementCount, testCase.elementCount);
        maxPartitionSize = std::max(maxPartitionSize,
                                    Buffers::partitionSize(testCase.workgroupSize, testCase.rows));
    }

    Buffers buffers = Buffers::allocate(context.alloc, maxElementCount, maxPartitionSize,
                                        merian::MemoryMappingType::NONE);

    Buffers stage = Buffers::allocate(context.alloc, maxElementCount, maxPartitionSize,
                                      merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    return std::make_tuple(buffers, stage);
}

using elem_t = float;
void uploadTestCase(const merian::CommandBufferHandle& cmd,
                    std::pmr::vector<elem_t> elements,
                    uint32_t workgroupSize,
                    uint32_t rows,
                    Buffers& buffers,
                    Buffers& stage) {

    std::size_t partitionSize = workgroupSize * rows;
    std::size_t workgroupCount = (elements.size() + partitionSize - 1) / partitionSize;

    SPDLOG_DEBUG("Staged upload");
    {
        Buffers::ElementsView stageView{stage.elements, elements.size()};
        Buffers::ElementsView localView{buffers.elements, elements.size()};
        stageView.template upload<elem_t>(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        Buffers::DecoupledStatesView localView{buffers.decoupledStates, workgroupCount};
        localView.zero(cmd);
        localView.expectComputeRead(cmd);
    }
}

void downloadToStage(const merian::CommandBufferHandle& cmd, Buffers& buffers, Buffers& stage) {
    Buffers::MeanView stageView{stage.mean};
    Buffers::MeanView localView{buffers.mean};
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

elem_t downloadFromStage(Buffers& stage) {
    Buffers::MeanView stageView{stage.mean};
    return stageView.template download<elem_t>();
}

bool runTestCase(const TestContext& context,
                 const TestCase& testCase,
                 Buffers& buffers,
                 Buffers& stage,
                 std::pmr::memory_resource* resource) {
    SPDLOG_INFO(fmt::format("Running test case for DecoupledMean:\n"
                            "\t-workgroupSize = {}\n"
                            "\t-rows = {}\n"
                            "\t-elementCount = {}\n"
                            "\t-distribution = {}\n"
                            "\t-stable = {}\n"
                            "\t-iterations = {}\n", //
                            testCase.workgroupSize, testCase.rows, testCase.elementCount,
                            wrs::distribution_to_pretty_string(testCase.distribution),
                            testCase.stable, testCase.iterations));
    SPDLOG_DEBUG("Creating DecoupledMean algorithm instance");
    wrs::DecoupledMean kernel(context.context, context.shaderCompiler, testCase.workgroupSize, testCase.rows,
                              testCase.stable);

    bool failed = false;
    for (size_t i = 0; i < testCase.iterations; ++i) {
        context.queue->wait_idle();

        if (testCase.iterations > 1) {
            if (testCase.elementCount > 5e5) {
                SPDLOG_INFO(
                    fmt::format("Testing iteration {} out of {}", i + 1, testCase.iterations));
            }
        }

        MERIAN_PROFILE_SCOPE(context.profiler,
                             fmt::format("TestCase: [workgroupSize={},rows={},elementCount={},"
                                         "distribution={},stable={}]",
                                         testCase.workgroupSize, testCase.rows,
                                         testCase.elementCount,
                                         wrs::distribution_to_pretty_string(testCase.distribution),
                                         testCase.stable));

        // Generate elements
        std::pmr::vector<elem_t> elements{resource};
        {
            SPDLOG_DEBUG(fmt::format("Generating {} weights with {}", testCase.elementCount,
                                     wrs::distribution_to_pretty_string(testCase.distribution)));
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            elements = wrs::pmr::generate_weights<elem_t>(testCase.distribution,
                                                          testCase.elementCount, resource);
        }

        // Begin recording
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();

        std::string recordingLabel = fmt::format(
            "Recoding: [workgroupSize={},rows={},elementCount={},"
            "distribution={},stable={},it={}]",
            testCase.workgroupSize, testCase.rows, testCase.elementCount,
            wrs::distribution_to_pretty_string(testCase.distribution), testCase.stable, i + 1);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // Upload elements
        {
            SPDLOG_DEBUG("Uploading elements");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Uploading elements");
            uploadTestCase(cmd, elements, testCase.workgroupSize, testCase.rows, buffers, stage);
        }

        // Run algorithm
        {
            SPDLOG_DEBUG("Running algorithm");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute DecoupledMean");
            kernel.run(cmd, buffers, testCase.elementCount);
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
        elem_t mean;
        {
            mean = downloadFromStage(stage);
        }

        // Compute reference
        elem_t referenceReduction = wrs::reference::tree_reduction<elem_t>(elements);
        elem_t referenceMean = referenceReduction / testCase.elementCount;

        if (std::abs(referenceMean - mean) > 0.01) {
            SPDLOG_ERROR(fmt::format("DecoupledMean is numerically unstable\n"
                                     "Expected {}, Got{}",
                                     referenceMean, mean));
            failed = true;
        }

        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::decoupled_mean::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing decoupled_mean algorithm");

    TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{buffers.elements->get_size() * 10};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, buffers, stage, resource);
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));
}
