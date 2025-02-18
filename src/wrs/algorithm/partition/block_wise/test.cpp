#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/test/is_partition.hpp"
#include "src/wrs/test/is_stable_partition.hpp"
#include "src/wrs/test/test.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>

#include "./BlockWisePartition.hpp"

namespace wrs::test::block_wise_partition {

using namespace wrs;
using namespace wrs::test;

using Algorithm = BlockWisePartition;
using Buffers = Algorithm::Buffers;

struct TestCase {
    BlockWisePartitionConfig config;
    glsl::uint N;
    Distribution dist;
    uint32_t iterations;
};

static const TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = BlockWisePartitionConfig(
            PartitionBlockScanConfig(512,
                                     2,
                                     2,
                                     BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE |
                                         BlockScanVariant::SUBGROUP_SCAN_INTRINSIC),
            BlockScanConfig(512,
                            2,
                            BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE |
                                BlockScanVariant::SUBGROUP_SCAN_SHFL,
                            2,
                            true),
            PartitionCombineConfig(512, 4, 1, 1)),
        .N = static_cast<glsl::uint>((1e6)),
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
        Buffers::BlockIndicesView stageView{stage.blockIndices, partitionCount};
        Buffers::BlockIndicesView localView{buffers.blockIndices, partitionCount};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
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
    std::pmr::vector<glsl::uint> indices;
    std::pmr::vector<glsl::uint> blockCount;

    std::pmr::vector<glsl::uint> partitionIndices;

    std::pmr::vector<float> partition;
    glsl::uint heavyCount;
};
static Results downloadFromStage(Buffers& stage,
                                 glsl::uint N,
                                 glsl::uint partitionCount,
                                 std::pmr::memory_resource* resource) {

    Buffers::IndicesView stagePrefixView{stage.indices, N};
    auto indices = stagePrefixView.download<glsl::uint, wrs::pmr_alloc<glsl::uint>>(resource);

    Buffers::BlockIndicesView stageReductionView{stage.blockIndices, partitionCount};
    auto blockCount = stageReductionView.download<glsl::uint, wrs::pmr_alloc<glsl::uint>>(resource);

    Buffers::HeavyCountView stageHeavyCountView{stage.heavyCount};
    glsl::uint heavyCount = stageHeavyCountView.download<glsl::uint>();

    Buffers::PartitionIndicesView stagePartiitonIndicesView{stage.partitionIndices, N};
    auto partitionIndices =
        stagePartiitonIndicesView.download<glsl::uint, wrs::pmr_alloc<glsl::uint>>(resource);

    Buffers::PartitionView stagePartitionView{stage.partition, N};
    auto partition = stagePartitionView.download<float, wrs::pmr_alloc<float>>(resource);

    return Results{
        .indices = std::move(indices),
        .blockCount = std::move(blockCount),
        .partitionIndices = std::move(partitionIndices),
        .partition = std::move(partition),
        .heavyCount = heavyCount,
    };
};

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName = fmt::format("{{blockSize={},N={}}}",
                                       testCase.config.elementScanConfig.blockSize(), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    if (testCase.config.maxElementCount() < testCase.N) {
        throw std::runtime_error(
            fmt::format("N={} is to large max={}", testCase.N, testCase.config.maxElementCount()));
    }

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

        const glsl::uint partitionCount =
            (testCase.N + testCase.config.blockSize() - 1) / testCase.config.blockSize();

        // 1. Generate input
        context.profiler->start("Generate test input");
        std::pmr::vector<float> elements =
            wrs::pmr::generate_weights<float>(testCase.dist, testCase.N, resource);
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
            kernel.run(cmd, buffers, testCase.N, context.profiler);
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
        cmd->end();
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
                fmt::println("BLOCK-WISE-PREFIX-SUMS:");
                for (std::size_t i = 0; i < results.indices.size(); ++i) {
                    fmt::println("[{}]: {} -> {}", i, elements[i] > pivot ? 1 : 0,
                                 results.indices[i]);
                }

                fmt::println("BLOCK-REDUCTIONS:");
                for (std::size_t i = 0; i < results.blockCount.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.blockCount[i]);
                }

                fmt::println("PARTITION-INDICES:");
                for (std::size_t i = 0; i < results.partitionIndices.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.partitionIndices[i]);
                }

                fmt::println("PARTITION:");
                for (std::size_t i = 0; i < results.partition.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.partition[i]);
                }
            }
            fmt::println("HeavyCount: {}", results.heavyCount);

            if (testCase.N < (1 << 24)) {
                const std::span<float> heavy{results.partition.begin(), results.heavyCount};

                const std::span<float> light{results.partition.begin() + results.heavyCount,
                                             testCase.N - results.heavyCount};
                std::reverse(light.begin(), light.end());

                auto err = wrs::test::pmr::assert_is_stable_partition<float>(heavy, light, elements,
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
} // namespace wrs::test::block_wise_partition
