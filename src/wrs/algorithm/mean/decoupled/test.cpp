#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/mean/decoupled/DecoupledMean.h"
#include "src/wrs/algorithm/mean/decoupled/test/test_cases.hpp"
#include "src/wrs/algorithm/mean/decoupled/test/test_setup.hpp"
#include "src/wrs/algorithm/mean/decoupled/test/test_types.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/test/test.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>

using namespace wrs::test;
using namespace wrs::test::decoupled_mean;

vk::DeviceSize wrs::test::decoupled_mean::sizeOfElement(const ElementType wt) {
    switch (wt) {
    case WEIGHT_TYPE_FLOAT:
        return sizeof(float);
    }
    throw std::runtime_error("NOT IMPLEMENTED");
}

template <typename elem_t>
void uploadTestCase(vk::CommandBuffer cmd,
                    std::pmr::vector<elem_t> elements,
                    uint32_t workgroupSize,
                    uint32_t rows,
                    Buffers& buffers,
                    Buffers& stage) {

    { // Upload elements
        elem_t* elementsMapped = stage.elements->get_memory()->map_as<elem_t>();
        std::memcpy(elementsMapped, elements.data(), sizeof(elem_t) * elements.size());
        stage.elements->get_memory()->unmap();
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                            {}, {},
                            stage.elements->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                           vk::AccessFlagBits::eTransferRead),
                            {});
        vk::BufferCopy copy{0, 0, elements.size() * sizeof(elem_t)};
        cmd.copyBuffer(*stage.elements, *buffers.elements, 1, &copy);

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                            buffers.elements->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                             vk::AccessFlagBits::eShaderRead),
                            {});
    }
    { // Reset decoupled state!

        vk::DeviceSize clearRange =
            Buffers::minDecoupledStateSize(elements.size(), workgroupSize, rows);
        // Round to the next multiple of 4 (see vulkan docs)
        clearRange = (clearRange + 4 - 1) / 4; // ceil div
        clearRange *= 4;

        if (clearRange >= buffers.decoupledStates->get_size()) {
            clearRange = VK_WHOLE_SIZE;
        }

        cmd.fillBuffer(*buffers.decoupledStates, 0, VK_WHOLE_SIZE, 0);

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {},
            buffers.decoupledStates->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                    vk::AccessFlagBits::eShaderRead),
            {});
    }
}

template <typename elem_t>
void downloadToStage(vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage) {
    {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {}, {},
                            buffers.mean->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                         vk::AccessFlagBits::eTransferRead),
                            {});
        vk::BufferCopy copy{0, 0, sizeof(elem_t)};
        cmd.copyBuffer(*buffers.mean, *stage.mean, 1, &copy);

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                            {}, {},
                            stage.mean->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                       vk::AccessFlagBits::eHostRead),
                            {});
    }
}

template <typename elem_t> elem_t downloadFromStage(Buffers& stage) {
    elem_t* meanMapped = stage.mean->get_memory()->map_as<elem_t>();
    return *meanMapped;
}

template <typename elem_t>
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
    wrs::DecoupledMean<elem_t> kernel(context.context, testCase.workgroupSize, testCase.rows,
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
            elements = std::move(wrs::pmr::generate_weights<elem_t>(
                testCase.distribution, testCase.elementCount, resource));
        }

        // Begin recording
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();

        std::string recordingLabel = fmt::format(
            "Recoding: [workgroupSize={},rows={},elementCount={},"
            "distribution={},stable={},it={}]",
            testCase.workgroupSize, testCase.rows, testCase.elementCount,
            wrs::distribution_to_pretty_string(testCase.distribution), testCase.stable, i + 1);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // Upload elements
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Uploading elements");
            uploadTestCase(cmd, elements, testCase.workgroupSize, testCase.rows, buffers, stage);
        }

        // Run algorithm
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute DecoupledMean");
            kernel.run(cmd, buffers, testCase.elementCount);
        }

        // Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download result to stage");
            downloadToStage<elem_t>(cmd, buffers, stage);
        }

        // Submit to queue
        {
            context.profiler->end();
            context.profiler->cmd_end(cmd);
            cmd.end();

            MERIAN_PROFILE_SCOPE(context.profiler, "Wait for device idle");
            context.queue->submit_wait(cmd);
        }

        // Download results from stage
        elem_t mean;
        {
            mean = downloadFromStage<elem_t>(stage);
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
    }
    return failed;
}

void wrs::test::decoupled_mean::test(const merian::ContextHandle& context) {

    TestContext testContext = setupTestContext(context);

    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{buffers.elements->get_size() * 10};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    for (const auto& testCase : TEST_CASES) {
        switch (testCase.elemType) {
        case WEIGHT_TYPE_FLOAT:
            runTestCase<float>(testContext, testCase, buffers, stage, resource);
            break;
        }
        stackResource.reset();
    }

    testContext.profiler->collect();
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));
}
