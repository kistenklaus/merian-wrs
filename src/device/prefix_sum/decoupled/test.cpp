#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/host/assert/is_prefix.hpp"
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
        auto weights = host::pmr::generate_weights(testCase.distribution, testCase.N);
        std::size_t partitionSize = testCase.config.partitionSize();
        // TODO
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
            uploadTestCase(cmd, buffers, stage, weights, partitionSize);
        }

        // 4. Run test case
        {
          host::glsl::uint workgroupCount = (testCase.N + testCase.config.partitionSize() - 1) /
                                        testCase.config.partitionSize();
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd,
                                     fmt::format("Execute algorithm [{}]", workgroupCount));
            SPDLOG_DEBUG("Execute algorithm ({})", workgroupCount);
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
            auto err = host::test::pmr::assert_is_inclusive_prefix<float>(weights, results.prefixSum,
                                                                         resource);

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
        context.profiler->collect(true, true);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing Decoupled prefix sum algorithm");

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

} // namespace device::test::decoupled_prefix
