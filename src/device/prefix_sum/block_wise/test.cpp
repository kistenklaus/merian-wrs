#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/host/assert/is_prefix.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/test/context.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <tuple>

#include "./BlockWiseScan.hpp"

namespace device::test::block_wise {

using Algorithm = BlockWiseScan;
using Buffers = Algorithm::Buffers;

struct TestCase {
    BlockWiseScanConfig config;
    host::glsl::uint N;
    host::Distribution dist;
    uint32_t iterations;
};

static const TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = BlockWiseScanConfig(
            BlockScanConfig(256, // workgroups size
                            2,   // rows
                            BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                            2, // sequential block scan length
                            true),
            BlockScanConfig(512,
                            4,
                            BlockScanVariant::RANKED | BlockScanVariant::SUBGROUP_SCAN_SHFL |
                                BlockScanVariant::EXCLUSIVE,
                            4,
                            false),
            BlockCombineConfig(512, 2, 1, 2)),
        .N = static_cast<host::glsl::uint>(1024 * 2048),
        .dist = host::Distribution::UNIFORM,
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
                           std::span<const float> elements) {
    Buffers::ElementsView stageView{stage.elements, elements.size()};
    Buffers::ElementsView localView{buffers.elements, elements.size()};
    stageView.upload(elements);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void downloadToStage(const merian::CommandBufferHandle& cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            host::glsl::uint N,
                            host::glsl::uint partitionCount) {
    {
        Buffers::PrefixSumView stageView{stage.prefixSum, N};
        Buffers::PrefixSumView localView{buffers.prefixSum, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::ReductionsView stageView{stage.reductions, partitionCount};
        Buffers::ReductionsView localView{buffers.reductions, partitionCount};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results {
    std::pmr::vector<float> reductions;
    std::pmr::vector<float> prefixSum;
};
static Results downloadFromStage(Buffers& stage,
                                 host::glsl::uint N,
                                 host::glsl::uint partitionCount,
                                 std::pmr::memory_resource* resource) {

    Buffers::PrefixSumView stagePrefixView{stage.prefixSum, N};
    auto prefixSums = stagePrefixView.download<float, host::pmr_alloc<float>>(resource);

    Buffers::ReductionsView stageReductionView{stage.reductions, partitionCount};
    auto reductions = stageReductionView.download<float, host::pmr_alloc<float>>(resource);

    return Results{
        .reductions = std::move(reductions),
        .prefixSum = std::move(prefixSums),
    };
};

static bool runTestCase(const host::test::TestContext& context,
                        const merian::ProfilerHandle& profiler,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{blockSize={},seq={},N={}}}", testCase.config.blockSize(),
                    testCase.config.elementScanConfig.sequentialScanLength, testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(profiler, testName);
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

        const host::glsl::uint partitionCount =
            (testCase.N + testCase.config.blockSize() - 1) / testCase.config.blockSize();

        // 1. Generate input
        profiler->start("Generate test input");
        std::pmr::vector<float> elements =
            host::pmr::generate_weights<float>(testCase.dist, testCase.N, resource);
        profiler->end();

        // 2. Begin recoding
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);
        profiler->start(recordingLabel);
        profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload test case indices
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Upload test case");
            SPDLOG_DEBUG("Uploading test case...");
            uploadTestCase(cmd, buffers, stage, elements);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N, profiler);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N, partitionCount);
        }

        // 6. Submit to device
        profiler->end();
        profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd->end();
        context.queue->submit_wait(cmd);

        // 7. Download from stage
        profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results results = downloadFromStage(stage, testCase.N, partitionCount, resource);
        profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");

            const auto err = host::test::pmr::assert_is_inclusive_prefix<float>(
                elements, results.prefixSum, resource);
            if (err) {
                fmt::println("Prefix sum is not perfect:\n{}", err.message());
            }

            if (testCase.N <= 1024) {
                fmt::println("PrefixSum:");
                for (std::size_t i = 0; i < results.prefixSum.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.prefixSum[i]);
                }
                fmt::println("BlockScan:");
                for (std::size_t i = 0; i < results.reductions.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.reductions[i]);
                }
            }
        }
        profiler->collect(true, true);
    }
    return failed;
}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing Work efficient prefix sum algorithm");

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(context);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context.context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context.context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    std::pmr::memory_resource* resource = std::pmr::get_default_resource();

    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, profiler, testCase, buffers, stage, resource);
    }

    profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(profiler->get_report())));
}

} // namespace device::test::block_wise
