#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/host/assert/is_prefix.hpp"
#include "src/host/gen/weight_generator.h"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "src/device/prefix_sum/PrefixSum.hpp"
#include "src/device/prefix_sum/test.hpp"
#include "src/host/test/context.hpp"

#ifndef MERIAN_PROFILER_ENABLE
#define MERIAN_PROFILER_ENABLE
#endif

namespace device::prefix_sum {

using base = host::glsl::f32;
using Algorithm = PrefixSum<base>;
using Buffers = Algorithm::Buffers;
using Config = Algorithm::Config;

struct TestCase {
    Config config;

    host::glsl::uint N;
    host::Distribution distribution;

    uint32_t iterations;
};

static TestCase TEST_CASES[] = {
    /* TestCase{ */
    /*     .config = DecoupledPrefixSumConfig( */
    /*         512, 4, BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_INTRINSIC, 32), */
    /*     .N = static_cast<host::glsl::uint>((1024 * 2048 + 1)), */
    /*     .distribution = host::Distribution::UNIFORM, */
    /*     .iterations = 5, */
    /* }, */
    /* TestCase{ */
    /*     .config = DecoupledPrefixSumConfig( */
    /*         512, 8, BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_INTRINSIC, 32), */
    /*     .N = static_cast<host::glsl::uint>((1024 * 2048 + 1)), */
    /*     .distribution = host::Distribution::UNIFORM, */
    /*     .iterations = 1, */
    /* }, */
    TestCase{
        .config = BlockWiseScanConfig(512, 8),
        .N = static_cast<host::glsl::uint>(1e4),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = BlockWiseScanConfig(512, 8),
        .N = static_cast<host::glsl::uint>(1e5),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = BlockWiseScanConfig(512, 16),
        .N = static_cast<host::glsl::uint>(1e6),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = BlockWiseScanConfig(512, 16),
        .N = static_cast<host::glsl::uint>(1e7),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = BlockWiseScanConfig(512, 16, 4),
        .N = static_cast<host::glsl::uint>(1e8),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },

    TestCase{
        .config = DecoupledPrefixSumConfig(),
        .N = static_cast<host::glsl::uint>((1e4)),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPrefixSumConfig(),
        .N = static_cast<host::glsl::uint>((1e5)),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPrefixSumConfig(),
        .N = static_cast<host::glsl::uint>((1e6)),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPrefixSumConfig(),
        .N = static_cast<host::glsl::uint>((1e7)),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPrefixSumConfig(),
        .N = static_cast<host::glsl::uint>((1e8)),
        .distribution = host::Distribution::UNIFORM,
        .iterations = 2,
    },
};

static void uploadTestCase(const merian::CommandBufferHandle cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> elements) {
    {
        Buffers::ElementsView<base> stageView{stage.elements, elements.size()};
        Buffers::ElementsView<base> localView{buffers.elements, elements.size()};
        stageView.upload(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
}

static void downloadToStage(const merian::CommandBufferHandle cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            std::size_t N) {
    Buffers::PrefixSumView<base> stageView{stage.prefixSum, N};
    Buffers::PrefixSumView<base> localView{buffers.prefixSum, N};
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

struct Results {
    std::pmr::vector<float> prefixSum;
};
static Results
downloadFromStage(Buffers& stage, std::size_t N, std::pmr::memory_resource* resource) {
    Buffers::PrefixSumView<base> stageView{stage.prefixSum, N};
    auto prefixSum = stageView.download<float, host::pmr_alloc<float>>(resource);

    return Results{
        .prefixSum = std::move(prefixSum),
    };
};

static void runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        std::pmr::memory_resource* resource) {

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context.context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context.context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(context.queue);

    Buffers buffers = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE,
                                        testCase.config, testCase.N);
    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      testCase.config, testCase.N);
    std::string testName =
        fmt::format("{{{},N={}}}", prefixSumConfigName(testCase.config), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    if (kernel.maxElementCount() < testCase.N) {
      throw std::runtime_error("Input size is to large");
    }

    std::string recordingLabel = fmt::format("Recording : {}", testName);

    host::test::TestResultType out = host::test::SUCCESS;
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
        profiler->end();

        // 2. Begin recoding
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();
        profiler->start(recordingLabel);
        profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload test case indices
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Upload test case");
            SPDLOG_DEBUG("Uploading test case...");
            uploadTestCase(cmd, buffers, stage, weights);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("Execute algorithm"));
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N, profiler);
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

            if (testCase.N <= 1024) {
                for (std::size_t i = 0;
                     i < std::min(results.prefixSum.size(), static_cast<std::size_t>(1024)); ++i) {
                    fmt::println("[{}] = {}", i, results.prefixSum[i]);
                }
            }
            auto err = host::test::pmr::assert_is_inclusive_prefix<float>(
                weights, results.prefixSum, resource);

            if (err) {
                SPDLOG_WARN("Invalid prefix: \n{}", err.message());
                out += host::test::WARNING;
            }
        }
        profiler->collect(true, true);
    }

    const auto report = profiler->get_report().gpu_report;
    auto recordingEntry =
        std::ranges::find_if(report, [&](const merian::Profiler::ReportEntry& entry) {
            return entry.name == recordingLabel;
        });
    if (recordingEntry == report.end()) {
        fmt::println("ENTRIES:");
        for (const auto& x : report) {
            fmt::println("{}", x.name);
        }
        throw std::runtime_error("Impossible state 1");
    }
    auto entry = std::ranges::find_if(recordingEntry->children,
                                      [&](const merian::Profiler::ReportEntry& entry) {
                                          return entry.name == "Execute algorithm";
                                      });
    if (entry == recordingEntry->children.end()) {
        throw std::runtime_error("Impossible state 2");
    }

    context.pushResult(
        "Scan", prefixSumConfigClass(testCase.config), out, entry->duration,
        entry->std_deviation,
        {host::test::TestProperty{.name = "N",
                                  .value = std::format("{:.0e}", static_cast<float>(testCase.N))}});
}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing Scan algorithm");

    std::pmr::memory_resource* resource = context.memory_resource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, testCase, resource);
    }
}

} // namespace device::prefix_sum
