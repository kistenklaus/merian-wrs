#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/renderdoc.hpp"
#include "src/wrs/algorithm/pack/simd/SimdPack.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_alias_table.hpp"
#include "src/wrs/test/is_split.hpp"
#include "src/wrs/test/test.hpp"
#include "src/wrs/types/alias_table.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory_resource>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vector>

#include "./SplitPack.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = SplitPack;
using Buffers = Algorithm::Buffers;

struct TestCase {
    glsl::uint workgroupSize;
    glsl::uint N;
    Distribution dist;
    glsl::uint splitSize;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .workgroupSize = 512,
        .N = 1024 * 2048,
        .dist = wrs::Distribution::PSEUDO_RANDOM_UNIFORM,
        .splitSize = 2,
        .iterations = 1,
    },
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    glsl::uint maxN = 0;
    for (const auto& testCase : TEST_CASES) {
        maxN = std::max(maxN, testCase.N);
    }

    Buffers stage =
        Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, maxN);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, maxN);

    return std::make_tuple(local, stage);
}

namespace wrs::test::splitpack {

void uploadTestCase(const vk::CommandBuffer cmd,
                    const Buffers& buffers,
                    const Buffers& stage,
                    std::span<const float> weights,
                    std::span<const glsl::uint> lightIndices,
                    std::span<const glsl::uint> heavyIndices,
                    std::span<const float> lightPrefix,
                    std::span<const float> heavyPrefix,
                    const float mean,
                    std::pmr::memory_resource* resource);

void downloadToStage(
    vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage, glsl::uint N, glsl::uint K);

Results
downloadFromStage(Buffers& stage, glsl::uint N, glsl::uint K, std::pmr::memory_resource* resource);

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{workgroupSize={},N={}}}", testCase.workgroupSize, testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, testCase.workgroupSize, testCase.splitSize};

    glsl::uint K = testCase.N / testCase.splitSize;

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
        const std::pmr::vector<float> weights =
            wrs::pmr::generate_weights<float>(testCase.dist, testCase.N);
        const float totalWeight = wrs::reference::kahan_reduction<float>(weights);
        const float averageWeight = totalWeight / static_cast<float>(testCase.N);
        const auto partitionIndices =
            wrs::reference::stable_partition_indicies<float, glsl::uint,
                                                      wrs::pmr_alloc<glsl::uint>>(
                weights, averageWeight, wrs::pmr_alloc<glsl::uint>(resource));
        const auto partition = wrs::reference::stable_partition<float, wrs::pmr_alloc<float>>(
            weights, averageWeight, wrs::pmr_alloc<float>(resource));
        const auto lightPrefix =
            wrs::reference::pmr::prefix_sum<float>(partition.light(), resource);
        const auto heavyPrefix =
            wrs::reference::pmr::prefix_sum<float>(partition.heavy(), resource);

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
            uploadTestCase(cmd, buffers, stage, weights, partitionIndices.light(),
                           partitionIndices.heavy(), lightPrefix, heavyPrefix, averageWeight,
                           resource);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N, K + 1);
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
        Results results = downloadFromStage(stage, testCase.N, K + 1, resource);
        context.profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");

            fmt::println("HEAVY-COUNT: {}", heavyPrefix.size());
            if (K <= 1024) {
                fmt::println("MEAN: {}", averageWeight);
                fmt::println("WEIGHTS");
                for (std::size_t i = 0; i < weights.size(); ++i) {
                  fmt::println("[{}]: {}", i, weights[i]);
                }
                fmt::println("PARTITION:");
                for (std::size_t i = 0; i < std::max(partitionIndices.light().size(), partitionIndices.heavy().size()); ++i) {
                    fmt::println("[{}]: {}    {}", i, i < partitionIndices.heavy().size() ? partitionIndices.heavy()[i] : -1,
                                 i < partitionIndices.light().size() ? partitionIndices.light()[i] : -1);
                }
                fmt::println("PREFIX:");
                for (std::size_t i = 0; i < std::max(lightPrefix.size(), heavyPrefix.size()); ++i) {
                    fmt::println("[{}]: {}    {}", i, i < heavyPrefix.size() ? heavyPrefix[i] : -1,
                                 i < lightPrefix.size() ? lightPrefix[i] : -1);
                }
                fmt::println("SPLITS");
                for (std::size_t i = 0; i < results.splits.size(); ++i) {
                    const auto& split = results.splits[i];
                    fmt::println("[{}]: ({},{},{})", i, split.i, split.j, split.spill);
                }

                fmt::println("ALIAS-TABLE:");
                for (std::size_t i = 0; i < results.aliasTable.size(); ++i) {
                    const auto& entry = results.aliasTable[i];
                    fmt::println("[{}]: ({},{})", i, entry.p, entry.a);
                }
            }

            /* auto err2 = wrs::test::pmr::assert_is_split<float,
             * glsl::uint>(std::span(results.splits.data() + 1, */
            /*       results.splits.size() - 1), K, heavyPrefix, lightPrefix, averageWeight,  */
            /*     0.01, */
            /*     resource); */
            /* if (err2) { */
            /*   SPDLOG_ERROR(err2.message()); */
            /* } */

            auto err = wrs::test::pmr::assert_is_alias_table<float, float, glsl::uint>(
                weights, results.aliasTable, totalWeight, 0.01, resource);
            if (err) {
                SPDLOG_ERROR(err.message());
            }
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing splitpack algorithm");

    const TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{4096 * 2048};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        renderdoc::startCapture();
        runTestCase(testContext, testCase, buffers, stage, resource);
        renderdoc::stopCapture();
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

} // namespace wrs::test::splitpack
