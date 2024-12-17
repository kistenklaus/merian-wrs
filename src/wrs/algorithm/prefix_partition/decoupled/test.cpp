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
#include "src/wrs/test/is_prefix.hpp"
#include "src/wrs/test/is_stable_partition.hpp"
#include "src/wrs/test/test.hpp"
#include "src/wrs/types/glsl.hpp"
#include "src/wrs/types/partition.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <memory_resource>
#include <ranges>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>
#include <src/wrs/reference/reduce.hpp>
#include <src/wrs/reference/split.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

using namespace wrs::test::decoupled_prefix_partition;

vk::DeviceSize wrs::test::decoupled_prefix_partition::sizeOfWeight(WeightT ty) {
    switch (ty) {
    case WEIGHT_T_FLOAT:
        return sizeof(float);
    }
    throw std::runtime_error("OH NO");
}

template <typename weight_t>
static void uploadTestCase(vk::CommandBuffer cmd,
                           std::span<const weight_t> elements,
                           std::size_t partitionSize,
                           weight_t pivot,
                           Buffers& buffers,
                           Buffers& stage) {
    {
        Buffers::ElementsView stageView{stage.elements, elements.size()};
        Buffers::ElementsView localView{buffers.elements, elements.size()};
        stageView.template upload<weight_t>(elements);
        stageView.copyTo(cmd, localView);
        stageView.expectComputeRead(cmd);
    }
    {
       Buffers::PivotView stageView{stage.pivot};
       Buffers::PivotView localView{buffers.pivot};
       stageView.template upload<weight_t>(pivot);
       stageView.copyTo(cmd, localView);
       localView.expectComputeRead(cmd);
    }
    {
      Buffers::BatchDescriptorsView localView{buffers.batchDescriptors, Buffers::workgroupCount(elements.size(), partitionSize)};
      localView.zero(cmd);
      localView.expectComputeRead(cmd);
    }
}

template <typename weight_t>
static void downloadResultsToStage(vk::CommandBuffer cmd,
                                   Buffers& buffers,
                                   Buffers& stage,
                                   bool writePartition,
                                   std::size_t elementCount) {
    if (writePartition) {
        Buffers::PartitionView stageView{stage.partition.value(), elementCount};
        Buffers::PartitionView localView{buffers.partition.value(), elementCount};

        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
      Buffers::PartitionPrefixView stageView{stage.partitionPrefix, elementCount};
      Buffers::PartitionPrefixView localView{buffers.partitionPrefix, elementCount};

      localView.expectComputeWrite();
      localView.copyTo(cmd, stageView);
      stageView.expectHostRead(cmd);
    }
}

template <typename weight_t>
static wrs::Partition<weight_t, std::pmr::vector<weight_t>>
downloadPrefixFromStage(Buffers& stage,
                        uint32_t elementCount,
                        std::pmr::memory_resource* resource) {
    
    Buffers::PartitionPrefixView stageView{stage.partitionPrefix, elementCount};
    wrs::glsl::uint heavyCount = stageView.attribute<"heavyCount">()
      .template download<wrs::glsl::uint>();
    std::pmr::vector<weight_t> storage = stageView.attribute<"heavyLightPrefix">()
      .template download<weight_t, wrs::pmr_alloc<weight_t>>(resource);
    
    wrs::Partition<weight_t, std::pmr::vector<weight_t>> heavyLightPrefix {std::move(storage), heavyCount};
    std::ranges::reverse(heavyLightPrefix.light());

    return heavyLightPrefix;
}

template <typename weight_t>
static wrs::Partition<wrs::glsl::uint, std::pmr::vector<wrs::glsl::uint>>
downloadPartitionFromStage(Buffers& stage,
                           uint32_t elementCount,
                           std::pmr::memory_resource* resource) {
    Buffers::PartitionView stageView{stage.partition.value(), elementCount};

    wrs::glsl::uint heavyCount = stageView.attribute<"heavyCount">().template download<wrs::glsl::uint>();

    std::pmr::vector<wrs::glsl::uint> storage = stageView.attribute<"heavyLightIndices">()
      .template download<wrs::glsl::uint, wrs::pmr_alloc<wrs::glsl::uint>>(resource);

    wrs::Partition<wrs::glsl::uint, std::pmr::vector<wrs::glsl::uint>> heavyLightPartition{std::move(storage),
                                                                              heavyCount};
    std::ranges::reverse(heavyLightPartition.light());
    return heavyLightPartition;
}

template <typename weight_t>
bool runTestCase(const wrs::test::TestContext& context,
                 Buffers& buffers,
                 Buffers& stage,
                 std::pmr::memory_resource* resource,
                 const TestCase& testCase) {
    SPDLOG_INFO(fmt::format("Running test case:\n\t-workgroupSize = "
                            "{}\n\t-rows={}\n\t-elementCount={}\n\t-distribution={}\n\t-stable={}"
                            "\n\t-writePartition={}\n\t-iterations={}",
                            testCase.workgroupSize, testCase.rows, testCase.elementCount,
                            wrs::distribution_to_pretty_string(testCase.distribution),
                            testCase.stable, testCase.writePartition, testCase.iterations));
    // 0. Create algorithm instance
    // NOTE: Allocators are not supported currently.
    SPDLOG_DEBUG("Creating DecoupledPrefixPartitionKernel");
    wrs::DecoupledPrefixPartition<weight_t> kernel(context.context, testCase.workgroupSize,
                                                   testCase.rows, testCase.writePartition,
                                                   testCase.stable);
    bool failed = false;
    for (size_t i = 0; i < testCase.iterations; ++i) {
        // Avoid side effects.
        context.queue->wait_idle();

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
            SPDLOG_DEBUG(fmt::format("Generating {} weights with {}", testCase.elementCount,
                                     wrs::distribution_to_pretty_string(testCase.distribution)));
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            elements = std::move(wrs::pmr::generate_weights<weight_t>(
                testCase.distribution, testCase.elementCount, resource));
        }
        //weight_t pivot = testCase.getPivot<weight_t>();
        auto pivot = wrs::reference::kahan_reduction<weight_t>(elements) / elements.size();

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
            SPDLOG_DEBUG("Uploading weights & pivot");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Uploading weights");
            std::size_t partitionSize = Buffers::partitionSize(testCase.workgroupSize, testCase.rows);
            uploadTestCase<weight_t>(cmd, elements, partitionSize, pivot, buffers, stage);
        }

        // 4. Run test case
        {
            SPDLOG_DEBUG("Executing kernel");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute kernel");
            kernel.run(cmd, buffers, testCase.elementCount);
        }

        // 5. Download results to stage
        {
            SPDLOG_DEBUG("Downloading results to staging buffers");
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download result to stage");
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
            context.queue->wait_idle();
        }

        // 7. Download from stage
        SPDLOG_DEBUG("Downloading results from staging buffers");

        std::pmr::vector<weight_t> _hack{resource};
        auto prefixTuple =
            std::make_tuple<std::span<weight_t>, std::span<weight_t>, std::pmr::vector<weight_t>>(
                {}, {}, std::move(_hack));

        wrs::Partition<weight_t, std::pmr::vector<weight_t>> heavyLightPrefix;

        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Downloading prefix from staging buffers");
            heavyLightPrefix = downloadPrefixFromStage<weight_t>(stage, testCase.elementCount, resource);
        }
        const auto heavyPrefix = heavyLightPrefix.heavy();
        const auto lightPrefix = heavyLightPrefix.light();

        wrs::Partition<uint32_t, std::pmr::vector<uint32_t>> heavyLightIndices;
        if (testCase.writePartition) {
            MERIAN_PROFILE_SCOPE(context.profiler, "Downloading partition from staging buffers");
            heavyLightIndices = downloadPartitionFromStage<weight_t>(stage, testCase.elementCount,
                                                                      resource);
        } else {
            MERIAN_PROFILE_SCOPE(context.profiler, "Compute reference stable partition");
            heavyLightIndices =
                wrs::reference::pmr::stable_partition_indicies<weight_t, uint32_t>(elements, pivot,
                                                                                   resource);
        }
        auto deref = [&](const auto i) { return elements[i]; };
        auto heavyPartitionView = heavyLightIndices.heavy() | std::views::transform(deref);
        std::pmr::vector<weight_t> heavyPartition{heavyPartitionView.begin(),
                                                  heavyPartitionView.end(), resource};
        auto lightPartitionView = heavyLightIndices.light() | std::views::transform(deref);
        std::pmr::vector<weight_t> lightPartition{lightPartitionView.begin(), lightPartitionView.end(), resource};


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
            const auto err = wrs::test::pmr::assert_is_inclusive_prefix<weight_t>(
                heavyPartition, heavyPrefix, resource);
            if (err) {
                SPDLOG_ERROR(fmt::format("Invalid heavy partition prefix!\n{}", err.message()));
                failed = true;
            }
        }
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Test light prefix");
            const auto err = wrs::test::pmr::assert_is_inclusive_prefix<weight_t>(
                lightPartition, lightPrefix, resource);
            if (err) {
                SPDLOG_ERROR(fmt::format("Invalid light partition prefix!\n{}", err.message()));
                failed = true;
            }
        }

        context.profiler->collect(true, true);

        {
            SPDLOG_DEBUG("Running experiment");
            wrs::reference::pmr::splitK<weight_t, wrs::glsl::uint>(heavyPrefix, lightPrefix, pivot, elements.size(), elements.size() / 32, resource);
        }
    }
    return failed;
}

void wrs::test::decoupled_prefix_partition::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing decoupled prefix partition algorithm");

    TestContext c = wrs::test::setupTestContext(context);

    auto [buffers, stage] = allocateBuffers(c);

    wrs::memory::StackResource stackResource{buffers.elements->get_size() * 10};
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
    c.profiler->collect(true,true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(c.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("decoupled prefix partition algorithm passed all tests");
    } else {
        SPDLOG_ERROR(fmt::format("decoupled prefix partition algorithm failed {} out of {} tests",
                                 failCount, sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}
