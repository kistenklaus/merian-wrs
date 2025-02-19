#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/assert/is_partition.hpp"
#include "src/host/assert/is_stable_partition.hpp"
#include "src/host/assert/test.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/memory/FallbackResource.hpp"
#include "src/host/memory/SafeResource.hpp"
#include "src/host/memory/StackResource.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include "src/device/partition/Partition.hpp"

namespace device::test::partition {

using base = host::glsl::f32;
using Algorithm = Partition<base>;
using Buffers = Algorithm::Buffers;
using Config = Algorithm::Config;

struct TestCase {
    Config config;
    host::glsl::uint N;
    host::Distribution dist;
    uint32_t iterations;
};

static const TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = BlockWisePartitionConfig(512, 2, 2, BlockScanVariant::RANKED),
        .N = static_cast<host::glsl::uint>(1024 * 2048 + 1),
        .dist = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPartitionConfig(512, 4, BlockScanVariant::RANKED_STRIDED),
        .N = static_cast<host::glsl::uint>(1024 * 2048 + 1),
        .dist = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 2,
    },
};

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const base> elements,
                           float pivot) {
    {
        Buffers::ElementsView<base> stageView{stage.elements, elements.size()};
        Buffers::ElementsView<base> localView{buffers.elements, elements.size()};
        stageView.upload(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        Buffers::PivotView<base> stageView{stage.pivot};
        Buffers::PivotView<base> localView{buffers.pivot};
        stageView.upload(pivot);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
}

static void downloadToStage(const merian::CommandBufferHandle& cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            host::glsl::uint N) {
    {
        Buffers::HeavyCountView stageView{stage.heavyCount};
        Buffers::HeavyCountView localView{buffers.heavyCount};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::PartitionIndicesView stageView{stage.partitionIndices, N};
        Buffers::PartitionIndicesView localView{buffers.partitionIndices, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::PartitionElementsView<base> stageView{stage.partitionElements, N};
        Buffers::PartitionElementsView<base> localView{buffers.partitionElements, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results {
    std::pmr::vector<host::glsl::uint> partitionIndices;
    std::pmr::vector<float> partitionElements;
    host::glsl::uint heavyCount;
};
static Results
downloadFromStage(Buffers& stage, host::glsl::uint N, std::pmr::memory_resource* resource) {

    Buffers::HeavyCountView stageHeavyCountView{stage.heavyCount};
    host::glsl::uint heavyCount = stageHeavyCountView.download<host::glsl::uint>();

    Buffers::PartitionIndicesView stagePartiitonIndicesView{stage.partitionIndices, N};
    auto partitionIndices =
        stagePartiitonIndicesView.download<host::glsl::uint, host::pmr_alloc<host::glsl::uint>>(
            resource);

    Buffers::PartitionElementsView<base> stagePartitionElementsView{stage.partitionElements, N};
    auto partition = stagePartitionElementsView.download<float, host::pmr_alloc<float>>(resource);

    return Results{
        .partitionIndices = std::move(partitionIndices),
        .partitionElements = std::move(partition),
        .heavyCount = heavyCount,
    };
};

static bool runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        std::pmr::memory_resource* resource) {

    Buffers buffers = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                        testCase.config, testCase.N);
    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      testCase.config, testCase.N);

    std::string testName =
        fmt::format("{{{},N={}}}", partitionConfigName(testCase.config), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    if (kernel.maxElementCount() < testCase.N) {
        throw std::runtime_error(
            fmt::format("N={} is to large max={}", testCase.N, kernel.maxElementCount()));
    }

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
        std::pmr::vector<float> elements =
            host::pmr::generate_weights<float>(testCase.dist, testCase.N, resource);
        const float pivot = 0.5;
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
            uploadTestCase(cmd, buffers, stage, elements, pivot);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, fmt::format("Execute algorithm"));
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N);
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

            if (testCase.N <= 1024) {

                fmt::println("PARTITION-INDICES:");
                for (std::size_t i = 0; i < results.partitionIndices.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.partitionIndices[i]);
                }

                fmt::println("PARTITION:");
                for (std::size_t i = 0; i < results.partitionElements.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.partitionElements[i]);
                }
            }
            fmt::println("HeavyCount: {}", results.heavyCount);

            if (testCase.N < (1 << 24)) {
                const std::span<float> heavy{results.partitionElements.begin(), results.heavyCount};

                const std::span<float> light{results.partitionElements.begin() + results.heavyCount,
                                             testCase.N - results.heavyCount};
                std::reverse(light.begin(), light.end());

                auto err = host::test::pmr::assert_is_stable_partition<float>(
                    heavy, light, elements, pivot, resource);
                if (err) {
                    SPDLOG_ERROR("{} constructs invalid partition: \n{}", testName, err.message());
                }
            }
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing partition algorithm");

    const host::test::TestContext testContext = host::test::setupTestContext(context);

    host::memory::StackResource stackResource{4096 * 2048};
    host::memory::FallbackResource fallbackResource{&stackResource};
    host::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, resource);
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results (Partition): \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("All tests passed");
    } else {
        SPDLOG_ERROR(fmt::format("Failed {} out of {} tests", failCount,
                                 sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}
} // namespace device::test::partition
