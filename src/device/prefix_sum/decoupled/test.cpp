#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/host/assert/is_prefix.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/test/context.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <tuple>

#include "./DecoupledPrefixSum.hpp"

namespace device::test::decoupled_prefix {

using Algorithm = DecoupledPrefixSum;
using Buffers = Algorithm::Buffers;

struct TestCase {
    DecoupledPrefixSumConfig config;

    host::glsl::uint N;
    host::Distribution distribution;

    uint32_t iterations;
};

static TestCase TEST_CASES[] = {
    TestCase{
        .config = DecoupledPrefixSumConfig(
            512, 8, BlockScanVariant::RANKED_STRIDED | BlockScanVariant::SUBGROUP_SCAN_SHFL),
        .N = static_cast<host::glsl::uint>((1 << 28)),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 1,
    },
};

std::tuple<Buffers, Buffers> allocateBuffers(const host::test::TestContext& context) {

    host::glsl::uint maxElementCount = 0;
    host::glsl::uint maxPartitionSize = 0;

    for (auto testCase : TEST_CASES) {
        maxElementCount = std::max(maxElementCount, testCase.N);
        host::glsl::uint partitionSize = testCase.config.partitionSize();
        maxPartitionSize = std::max(maxPartitionSize, partitionSize);
    }
    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxElementCount, maxPartitionSize);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                      maxElementCount, maxPartitionSize);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const merian::CommandBufferHandle cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> elements,
                           std::size_t partitionSize) {
    {
        Buffers::ElementsView stageView{stage.elements, elements.size()};
        Buffers::ElementsView localView{buffers.elements, elements.size()};
        stageView.upload(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        std::size_t partitionCount = (elements.size() + partitionSize - 1) / partitionSize;
        Buffers::DecoupledStatesView localView{buffers.decoupledStates, partitionCount};
        localView.zero(cmd);
        localView.expectComputeRead(cmd);
    }
}

static void downloadToStage(const merian::CommandBufferHandle cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            std::size_t N) {
    Buffers::PrefixSumView stageView{stage.prefixSum, N};
    Buffers::PrefixSumView localView{buffers.prefixSum, N};
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

struct Results {
    std::pmr::vector<float> prefixSum;
};
static Results
downloadFromStage(Buffers& stage, std::size_t N, std::pmr::memory_resource* resource) {
    Buffers::PrefixSumView stageView{stage.prefixSum, N};
    auto prefixSum = stageView.download<float, host::pmr_alloc<float>>(resource);

    return Results{
        .prefixSum = std::move(prefixSum),
    };
};

static bool runTestCase(const host::test::TestContext& context,
                        const merian::ProfilerHandle& profiler,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{workgroupSize={},N={},rows={},lookback={}}}", testCase.config.workgroupSize,
                    testCase.N, testCase.config.rows, testCase.config.parallelLookbackDepth);
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

        // 1. Generate input
        profiler->start("Generate test input");
        auto weights = host::pmr::generate_weights(testCase.distribution, testCase.N);
        std::size_t partitionSize = testCase.config.partitionSize();
        // TODO
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
            uploadTestCase(cmd, buffers, stage, weights, partitionSize);
        }

        // 4. Run test case
        {
            host::glsl::uint workgroupCount = (testCase.N + testCase.config.partitionSize() - 1) /
                                              testCase.config.partitionSize();
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd,
                                     fmt::format("Execute algorithm [{}]", workgroupCount));
            SPDLOG_DEBUG("Execute algorithm ({})", workgroupCount);
            kernel.run(cmd, buffers, testCase.N);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N);
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
        Results results = downloadFromStage(stage, testCase.N, resource);
        profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");
            auto err = host::test::pmr::assert_is_inclusive_prefix<float>(
                weights, results.prefixSum, resource);

            if (testCase.N <= 1024 * 2048) {
                for (std::size_t i = 0;
                     i < std::min(results.prefixSum.size(), static_cast<std::size_t>(1024)); ++i) {
                    fmt::println("[{}] = {}", i, results.prefixSum[i]);
                }
            }

            if (err) {
                SPDLOG_ERROR("Invalid prefix: \n{}", err.message());
            }
        }
        profiler->collect(true, true);
    }
    return failed;
}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing Decoupled prefix sum algorithm");

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(context);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context.context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context.context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    std::pmr::memory_resource* resource = context.memory_resource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, profiler, testCase, buffers, stage, resource);
    }

    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(profiler->get_report())));
}

} // namespace device::test::decoupled_prefix
