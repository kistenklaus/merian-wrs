#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_alias_table.hpp"
#include "src/wrs/test/is_prefix.hpp"
#include "src/wrs/test/test.hpp"
#include "src/wrs/types/alias_table.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tuple>

#include "./PartitionBlockScan.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = PartitionBlockScan;
using Buffers = Algorithm::Buffers;

struct TestCase {
    PartitionBlockScanConfig config;
    glsl::uint N;
    Distribution dist;
    uint32_t iterations;
};

static constexpr BlockScanVariant VARIANT = BlockScanVariant::RANKED_STRIDED | BlockScanVariant::SUBGROUP_SCAN_INTRINSIC
    | BlockScanVariant::EXCLUSIVE;

static const TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = PartitionBlockScanConfig(512, 1, 1, VARIANT),
        .N = static_cast<glsl::uint>((1 << 10)),
        .dist = wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 1,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    glsl::uint maxN = 0;
    glsl::uint maxPartitionCount = 0;
    for (const auto& testCase : TEST_CASES) {
        maxN = std::max(maxN, testCase.N);
        glsl::uint partitionCount =
            (testCase.N + testCase.config.blockSize() - 1) / testCase.config.blockSize();
        maxPartitionCount = std::max(maxPartitionCount, partitionCount);
    }

    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      maxN, maxPartitionCount);
    Buffers local =
        Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, maxN, maxPartitionCount);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const vk::CommandBuffer cmd,
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

static void downloadToStage(vk::CommandBuffer cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            glsl::uint N,
                            glsl::uint partitionCount) {
    {
        Buffers::IndicesView stageView{stage.indices, N};
        Buffers::IndicesView localView{buffers.indices, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::BlockCountView stageView{stage.blockCount, partitionCount};
        Buffers::BlockCountView localView{buffers.blockCount, partitionCount};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results {
    std::pmr::vector<glsl::uint> indices;
    std::pmr::vector<glsl::uint> blockCount;
};
static Results downloadFromStage(Buffers& stage,
                                 glsl::uint N,
                                 glsl::uint partitionCount,
                                 std::pmr::memory_resource* resource) {

    Buffers::IndicesView stagePrefixView{stage.indices, N};
    auto indices = stagePrefixView.download<glsl::uint, wrs::pmr_alloc<glsl::uint>>(resource);

    Buffers::BlockCountView stageReductionView{stage.blockCount, partitionCount};
    auto blockCount = stageReductionView.download<glsl::uint, wrs::pmr_alloc<glsl::uint>>(resource);

    return Results{
        .indices = std::move(indices),
        .blockCount = std::move(blockCount),
    };
};

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{workgroupSize={},rows={},seq={},N={}}}", testCase.config.workgroupSize,
                    testCase.config.rows, testCase.config.sequentialScanLength, testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, testCase.config};

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

        const glsl::uint partitionCount =
            (testCase.N + testCase.config.blockSize() - 1) / testCase.config.blockSize();

        // 1. Generate input
        context.profiler->start("Generate test input");
        std::pmr::vector<float> elements =
            wrs::pmr::generate_weights<float>(testCase.dist, testCase.N, resource);
        const float pivot = 0.5;
        context.profiler->end();

        // 2. Begin recoding
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
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
            glsl::uint workgroupCount = (testCase.N + kernel.blockSize() - 1) / kernel.blockSize();
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd,
                                     fmt::format("Execute algorithm [{}]", workgroupCount));
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N, partitionCount);
        }

        // 6. Submit to device
        context.profiler->end();
        context.profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd.end();
        context.queue->submit_wait(cmd);

        // 7. Download from stage
        context.profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results results = downloadFromStage(stage, testCase.N, partitionCount, resource);
        context.profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");

            if (testCase.N <= 1024) {
                fmt::println("INDICES:");
                uint h = 0;
                for (std::size_t i = 0; i < results.indices.size(); ++i) {
                    bool heavy = elements[i] > pivot;
                    if (heavy) {
                      if (results.indices[i] != h) {
                        fmt::println("ERROR");
                      }
                      h += 1;
                    }

                    fmt::println("[{}]: {} -> {}", i, elements[i] > pivot ? 1 : 0,
                                 results.indices[i]);
                }

                fmt::println("BLOCK-COUNTS:");
                for (std::size_t i = 0; i < results.blockCount.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.blockCount[i]);
                }
            }
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::partition::block_scan::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing Work efficient prefix sum algorithm");

    const TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{4096 * 2048};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

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
