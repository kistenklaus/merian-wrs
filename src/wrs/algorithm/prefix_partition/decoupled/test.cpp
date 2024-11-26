#include "./test.hpp"
#include "./test/test_setup.h"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/DecoupledPrefixPartitionKernel.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test/test_types.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/test/is_prefix.hpp"
#include "src/wrs/test/is_stable_partition.hpp"
#include "src/wrs/test/test.hpp"
#include <algorithm>
#include <cstring>
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

using namespace wrs::test::decoupled_prefix_partition;

template <typename weight_t>
static void uploadTestCase(vk::CommandBuffer cmd,
                           const std::span<weight_t> elements,
                           weight_t pivot,
                           Buffers& buffers,
                           Buffers& stage) {
    {
        weight_t* elementsMapped = stage.elements->get_memory()->map_as<weight_t>();
        std::memcpy(elementsMapped, elements.data(), elements.size() * sizeof(weight_t));
        stage.elements->get_memory()->unmap();
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                            {}, {},
                            stage.elements->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                           vk::AccessFlagBits::eTransferRead),
                            {});
        vk::BufferCopy copy{0, 0, elements.size() * sizeof(weight_t)};
        cmd.copyBuffer(*stage.elements, *buffers.elements, 1, &copy);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                            buffers.elements->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                             vk::AccessFlagBits::eShaderRead),
                            {});
    }
    {
        weight_t* pivotMapped = stage.pivot->get_memory()->map_as<weight_t>();
        std::memcpy(pivotMapped, &pivot, sizeof(weight_t));
        stage.pivot->get_memory()->unmap();
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                            {}, {},
                            stage.pivot->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                        vk::AccessFlagBits::eTransferRead),
                            {});
        vk::BufferCopy copy{0, 0, sizeof(weight_t)};
        cmd.copyBuffer(*stage.pivot, *buffers.pivot, 1, &copy);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                            buffers.pivot->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                          vk::AccessFlagBits::eShaderRead),
                            {});
    }
    {
        cmd.fillBuffer(*buffers.batchDescriptors, 0, buffers.batchDescriptors->get_size(), 0);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {},
            buffers.batchDescriptors->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                     vk::AccessFlagBits::eShaderRead),
            {});
    }
}

template <typename weight_t>
static void downloadResultsToStage(vk::CommandBuffer cmd,
                                   Buffers& buffers,
                                   Buffers& stage,
                                   bool writePartition,
                                   uint32_t elementCount) {
    if (writePartition) {
        vk::BufferCopy copy{0, 0, elementCount * sizeof(weight_t)};
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
            buffers.partition.value()->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eTransferRead),
            {});
        cmd.copyBuffer(*buffers.partition.value(), *stage.partition.value(), 1, &copy);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                            {}, {},
                            stage.partition.value()->buffer_barrier(
                                vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead),
                            {});
    }
    {
        vk::BufferCopy copy{0, 0, elementCount * sizeof(weight_t) + sizeof(uint32_t)};
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
            buffers.partitionPrefix->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                    vk::AccessFlagBits::eTransferRead),
            {});
        cmd.copyBuffer(*buffers.partitionPrefix, *stage.partitionPrefix, 1, &copy);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                            {}, {},
                            stage.partitionPrefix->buffer_barrier(
                                vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead),
                            {});
    }
}

template <typename weight_t>
static std::tuple<std::span<weight_t>, std::span<weight_t>, std::pmr::vector<weight_t>>
downloadPrefixFromStage(Buffers& stage,
                        uint32_t elementCount,
                        std::pmr::memory_resource* resource) {
    std::byte* mapped = stage.partitionPrefix->get_memory()->map_as<std::byte>();
    uint32_t heavyCount = *reinterpret_cast<uint32_t*>(mapped);
    // NOTE: This assumes that no alignment is required, which might be incorrect for
    // sizeof(weight_t) = 8 (e.g double)
    weight_t* heavyLightMapped = reinterpret_cast<weight_t*>(mapped + sizeof(uint32_t));
    std::pmr::vector<weight_t> storage{elementCount, resource};
    std::memcpy(storage.data(), heavyLightMapped, elementCount * sizeof(weight_t));
    stage.partitionPrefix->get_memory()->unmap();
    std::span<weight_t> heavy{storage.begin(), storage.begin() + heavyCount};
    std::span<weight_t> light{storage.begin() + heavyCount, storage.end()};
    std::reverse(light.begin(), light.end());

    return std::make_tuple(heavy, light, std::move(storage));
}

template <typename weight_t>
static std::tuple<std::span<weight_t>, std::span<weight_t>, std::pmr::vector<weight_t>>
downloadPartitionFromStage(Buffers& stage,
                           uint32_t elementCount,
                           uint32_t heavyCount,
                           std::pmr::memory_resource* resource) {
    weight_t* partitionMapped = stage.partition.value()->get_memory()->map_as<weight_t>();
    std::pmr::vector<weight_t> storage{elementCount, resource};
    std::memcpy(storage.data(), partitionMapped, elementCount * sizeof(weight_t));
    stage.partition.value()->get_memory()->unmap();
    std::span<weight_t> heavy{storage.begin(), storage.begin() + heavyCount};
    std::span<weight_t> light{storage.begin() + heavyCount, storage.end()};
    std::reverse(light.begin(), light.end());
    return std::make_tuple(heavy, light, std::move(storage));
}

template <typename weight_t>
bool runTestCase(const wrs::test::TestContext& context,
                 Buffers& buffers,
                 Buffers& stage,
                 std::pmr::memory_resource* resource,
                 const TestCase& testCase) {
    // Avoid side effects.
    context.queue->wait_idle();
    SPDLOG_INFO(fmt::format("Running test case:\n\t-workgroupSize = "
                            "{}\n\t-rows={}\n\t-elementCount={}\n\t-distribution={}\n\t-stable={}"
                            "\n\t-writePartition={}\n\t-iterations={}",
                            testCase.workgroupSize, testCase.rows, testCase.elementCount,
                            wrs::distribution_to_pretty_string(testCase.distribution),
                            testCase.stable, testCase.writePartition, testCase.iterations));
    // 0. Create algorithm instance
    // NOTE: Allocators are not supported currently.
    SPDLOG_DEBUG("Creating DecoupledPrefixPartitionKernel");
    wrs::DecoupledPrefixPartitionKernel<weight_t> kernel(context.context, testCase.workgroupSize,
                                                         testCase.rows, testCase.writePartition,
                                                         testCase.stable);
    bool failed = false;
    for (size_t i = 0; i < testCase.iterations; ++i) {
        if (testCase.iterations > 1) {
            if (testCase.elementCount > 1e6) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", i + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", i + 1, testCase.iterations));
            }
        }
        MERIAN_PROFILE_SCOPE(context.profiler,
                             fmt::format("TestCase: [workgroupSize={},rows={},elementCount={},"
                                         "distribution={},stable={},writePartition={}]",
                                         testCase.workgroupSize, testCase.rows,
                                         testCase.elementCount,
                                         wrs::distribution_to_pretty_string(testCase.distribution),
                                         testCase.stable, testCase.writePartition));

        // 1. Generate weights
        std::pmr::vector<weight_t> elements{resource};
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            SPDLOG_DEBUG(fmt::format("Generating {} weights with {}", testCase.elementCount,
                                     wrs::distribution_to_pretty_string(testCase.distribution)));
            elements = std::move(wrs::pmr::generate_weights<weight_t>(
                testCase.distribution, testCase.elementCount, resource));
        }
        weight_t pivot = testCase.getPivot<weight_t>();

        // 2. Begin recording
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
        std::string recordingLabel =
            fmt::format("Recoding: [workgroupSize={},rows={},elementCount={},"
                        "distribution={},stable={},writePartition={}]",
                        testCase.workgroupSize, testCase.rows, testCase.elementCount,
                        wrs::distribution_to_pretty_string(testCase.distribution), testCase.stable,
                        testCase.writePartition);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload weights & pivot & reset descriptor states
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Uploading weights");
            SPDLOG_DEBUG("Uploading weights & pivot");
            uploadTestCase<weight_t>(cmd, elements, pivot, buffers, stage);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute kernel");
            SPDLOG_DEBUG("Executing kernel");
            kernel.run(cmd, buffers, testCase.elementCount);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download result to stage");
            SPDLOG_DEBUG("Downloading results to staging buffers");
            downloadResultsToStage<weight_t>(cmd, buffers, stage, testCase.writePartition,
                                             testCase.elementCount);
        }

        context.profiler->end();        // end recoding
        context.profiler->cmd_end(cmd); // end recoding

        // 6. Submit to device
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Wait for device idle");
            cmd.end();
            context.queue->submit_wait(cmd);
        }

        // 7. Download from stage
        SPDLOG_DEBUG("Downloading results from staging buffers");

        std::pmr::vector<weight_t> _hack{resource};
        auto prefixTuple =
            std::make_tuple<std::span<weight_t>, std::span<weight_t>, std::pmr::vector<weight_t>>(
                {}, {}, std::move(_hack));

        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Downloading prefix from staging buffers");
            auto temp = downloadPrefixFromStage<weight_t>(stage, testCase.elementCount, resource);
            std::swap(temp, prefixTuple);
        }
        const auto& [heavyPrefix, lightPrefix, prefixStorage] = prefixTuple;

        std::span<weight_t> heavyPartition;
        std::span<weight_t> lightPartition;
        std::pmr::vector<weight_t> partitionStorage{resource};
        if (testCase.writePartition) {
            MERIAN_PROFILE_SCOPE(context.profiler, "Downloading partition from staging buffers");
            auto [heavy, light, storage] = downloadPartitionFromStage<weight_t>(
                stage, testCase.elementCount, heavyPrefix.size(), resource);
            partitionStorage.swap(storage);
            heavyPartition = heavy;
            lightPartition = light;
        } else {
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute reference stable partition");
            auto [heavy, light, storage] =
                wrs::reference::pmr::stable_partition<weight_t>(elements, pivot, resource);
            partitionStorage.swap(storage);
            heavyPartition = heavy;
            lightPartition = light;
        }

        SPDLOG_DEBUG("Testing partition");
        // 8. Test partition
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Test light heavy partition");
            const auto err = wrs::test::pmr::assert_is_stable_partition<weight_t>(
                heavyPartition, lightPartition, elements, pivot, resource);
            if (err) {
                SPDLOG_ERROR(fmt::format("Invalid heavy/-light partition!\n{}", err.message()));
                failed = true;
            }
        }

        SPDLOG_DEBUG("Testing prefix scan");
        // 9. Test prefix sums
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Test heavy prefix");
            const auto err =
                wrs::test::pmr::assert_is_inclusive_prefix(heavyPartition, heavyPrefix, resource);
            if (err) {
                SPDLOG_ERROR(fmt::format("Invalid heavy partition prefix!\n{}", err.message()));
                failed = true;
            }
        }
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Test light prefix");
            const auto err =
                wrs::test::pmr::assert_is_inclusive_prefix(lightPartition, lightPrefix, resource);
            if (err) {
                SPDLOG_ERROR(fmt::format("Invalid light partition prefix!\n{}", err.message()));
                failed = true;
            }
        }
    }
    return failed;
}

void wrs::test::decoupled_prefix_partition::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing decoupled prefix partition algorithm");

    TestContext c = wrs::test::setupTestContext(context);

    auto [buffers, stage] = allocateBuffers(c);

    wrs::memory::StackResource stackResource{buffers.elements->get_size() * 1000};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        switch (testCase.weight_type) {
        case WEIGHT_T_FLOAT:
            if (runTestCase<float>(c, buffers, stage, resource, testCase)) {
                failCount += 1;
            }
            break;
        }
        stackResource.reset();
    }
    c.profiler->collect();
    SPDLOG_DEBUG(fmt::format("Profiler results: \n{}",
                             merian::Profiler::get_report_str(c.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("decoupled prefix partition algorithm passed all tests");
    } else {
        SPDLOG_ERROR(fmt::format("decoupled prefix partition algorithm failed {} out of {} tests",
                                 failCount, sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}
