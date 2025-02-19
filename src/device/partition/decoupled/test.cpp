#include "./test.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/utils/profiler.hpp"
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
#include <tuple>

#include "./DecoupledPartition.hpp"

namespace device::test::decoupled_partition {

using Algorithm = DecoupledPartition;
using Buffers = Algorithm::Buffers;

struct TestCase {
    DecoupledPartitionConfig config;
    host::glsl::uint N;
    host::Distribution dist;
    uint32_t iterations;
};

static const TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = DecoupledPartitionConfig(512,
                                           16,
                                           BlockScanVariant::RANKED_STRIDED |
                                               BlockScanVariant::EXCLUSIVE |
                                               BlockScanVariant::SUBGROUP_SCAN_INTRINSIC,
                                           32),
        .N = static_cast<host::glsl::uint>((1e8)),
        .dist = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 1,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const host::test::TestContext& context) {
    host::glsl::uint maxN = 0;
    host::glsl::uint maxPartitionCount = 0;
    for (const auto& testCase : TEST_CASES) {
        maxN = std::max(maxN, testCase.N);
        host::glsl::uint partitionCount =
            (testCase.N + testCase.config.blockSize() - 1) / testCase.config.blockSize();
        maxPartitionCount = std::max(maxPartitionCount, partitionCount);
    }

    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxN, maxPartitionCount);
    Buffers local =
        Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, maxN, maxPartitionCount);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> elements,
                           float pivot) {
    {
        Buffers::ElementsView stageView{stage.elements, elements.size()};
        Buffers::ElementsView localView{buffers.elements, elements.size()};
        stageView.upload(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        Buffers::PivotView stageView{stage.pivot};
        Buffers::PivotView localView{buffers.pivot};
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
        Buffers::PartitionView stageView{stage.partition, N};
        Buffers::PartitionView localView{buffers.partition, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results {
    std::pmr::vector<host::glsl::uint> partitionIndices;
    std::pmr::vector<float> partition;
    host::glsl::uint heavyCount;
};
static Results
downloadFromStage(Buffers& stage, host::glsl::uint N, std::pmr::memory_resource* resource) {

    Buffers::HeavyCountView stageHeavyCountView{stage.heavyCount};
    host::glsl::uint heavyCount = stageHeavyCountView.download<host::glsl::uint>();

    Buffers::PartitionIndicesView stagePartiitonIndicesView{stage.partitionIndices, N};
    auto partitionIndices =
        stagePartiitonIndicesView.download<host::glsl::uint, host::pmr_alloc<host::glsl::uint>>(resource);

    Buffers::PartitionView stagePartitionView{stage.partition, N};
    auto partition = stagePartitionView.download<float, host::pmr_alloc<float>>(resource);

    return Results{
        .partitionIndices = std::move(partitionIndices),
        .partition = std::move(partition),
        .heavyCount = heavyCount,
    };
};

static bool runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{blockSize={},N={}}}", testCase.config.blockSize(), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

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
          host::glsl::uint blockCount = (testCase.N + kernel.blockSize() - 1) / kernel.blockSize();
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd,
                                     fmt::format("Execute algorithm {}", blockCount));
            SPDLOG_DEBUG("Execute algorithm {}", blockCount);
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
            }
            fmt::println("HeavyCount: {}", results.heavyCount);

            if (testCase.N < (1 << 24)) {
                const std::span<float> heavy{results.partition.begin(), results.heavyCount};

                const std::span<float> light{results.partition.begin() + results.heavyCount,
                                             testCase.N - results.heavyCount};
                std::reverse(light.begin(), light.end());

                auto err = host::test::pmr::assert_is_stable_partition<float>(heavy, light, elements,
                                                                             pivot, resource);
                if (err) {
                    SPDLOG_ERROR("Invalid partition: \n{}", err.message());
                }
            }
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing Work efficient prefix sum algorithm");

    const host::test::TestContext testContext = host::test::setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    host::memory::StackResource stackResource{4096 * 2048};
    host::memory::FallbackResource fallbackResource{&stackResource};
    host::memory::SafeResource safeResource{&fallbackResource};

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

} // namespace device::test::decoupled_partition
