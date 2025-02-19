#include "./test.hpp"
/**
 * @author      : kistenklaus (karlsasssie@gmail.com)
 * @created     : 11/02/2025
 * @filename    : test.cpp
 */

#include "merian/vk/utils/profiler.hpp"
#include "src/device/mean/decoupled/DecoupledMean.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/memory/FallbackResource.hpp"
#include "src/host/memory/SafeResource.hpp"
#include "src/host/memory/StackResource.hpp"
#include "src/host/reference/reduce.hpp"
#include "src/host/assert/test.hpp"
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>

using namespace device;
using namespace host;
using namespace host::test;

namespace device::test::decoupled_mean {

using Buffers = device::DecoupledMeanBuffers;

struct TestCase {
    DecoupledMeanConfig config;
    uint32_t elementCount;
    Distribution distribution;
    bool stable;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = {},
        .elementCount = static_cast<uint32_t>(1024 * 2048),
        .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
        .stable = false,
        .iterations = 1,
    },
};

inline std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {

    uint32_t maxElementCount = 0;
    std::size_t maxPartitionSize = 0;

    for (const auto& testCase : TEST_CASES) {
        maxElementCount = std::max(maxElementCount, testCase.elementCount);
        maxPartitionSize =
            std::max(maxPartitionSize,
                     Buffers::partitionSize(testCase.config.workgroupSize, testCase.config.rows));
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
                            testCase.config.workgroupSize, testCase.config.rows,
                            testCase.elementCount,
                            host::distribution_to_pretty_string(testCase.distribution),
                            testCase.stable, testCase.iterations));
    SPDLOG_DEBUG("Creating DecoupledMean algorithm instance");
    DecoupledMean kernel(context.context, context.shaderCompiler, testCase.config);

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
                                         testCase.config.workgroupSize, testCase.config.rows,
                                         testCase.elementCount,
                                         host::distribution_to_pretty_string(testCase.distribution),
                                         testCase.stable));

        // Generate elements
        std::pmr::vector<elem_t> elements{resource};
        {
            SPDLOG_DEBUG(fmt::format("Generating {} weights with {}", testCase.elementCount,
                                     host::distribution_to_pretty_string(testCase.distribution)));
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            elements = host::pmr::generate_weights<elem_t>(testCase.distribution,
                                                          testCase.elementCount, resource);
        }

        // Begin recording
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();

        std::string recordingLabel = fmt::format(
            "Recoding: [workgroupSize={},rows={},elementCount={},"
            "distribution={},stable={},it={}]",
            testCase.config.workgroupSize, testCase.config.rows, testCase.elementCount,
            host::distribution_to_pretty_string(testCase.distribution), testCase.stable, i + 1);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // Upload elements
        {
            SPDLOG_DEBUG("Uploading elements");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Uploading elements");
            uploadTestCase(cmd, elements, testCase.config.workgroupSize, testCase.config.rows,
                           buffers, stage);
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
        elem_t referenceReduction = host::reference::reduce<elem_t>(elements);
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

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing decoupled_mean algorithm");

    TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    host::memory::StackResource stackResource{buffers.elements->get_size() * 10};
    host::memory::FallbackResource fallbackResource{&stackResource};
    host::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, buffers, stage, resource);
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));
}

} // namespace device::test::decoupled_mean
