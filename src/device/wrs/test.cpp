#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/statistics/chi_square/ChiSquare.hpp"
#include "src/device/statistics/chi_square/ChiSquareAllocFlags.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/test/context.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <random>
#include <spdlog/spdlog.h>

#include "src/device/wrs/WRS.hpp"
#include "src/host/statistics/js_divergence.hpp"
#include "src/host/why.hpp"
#include "vulkan/vulkan_enums.hpp"

#ifndef MERIAN_PROFILER_ENABLE
#define MERIAN_PROFILER_ENABLE
#endif

namespace device::wrs {

using Algorithm = WRS;
using Buffers = Algorithm::Buffers;
using Config = Algorithm::Config;

struct TestCase {
    Config config;
    host::glsl::uint N;
    host::Distribution distribution;
    host::glsl::uint S;
    uint32_t iterations;
};

static const TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = ITSConfig(DecoupledPrefixSumConfig(), InverseTransformSamplingConfig(512, 0)),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = ITSConfig(DecoupledPrefixSumConfig(), InverseTransformSamplingConfig(512, 0)),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = ITSConfig(DecoupledPrefixSumConfig(), InverseTransformSamplingConfig(512, 0)),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },

    TestCase{
        .config = ITSConfig(DecoupledPrefixSumConfig(), InverseTransformSamplingConfig(512, 32)),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = ITSConfig(DecoupledPrefixSumConfig(), InverseTransformSamplingConfig(512, 32)),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = ITSConfig(DecoupledPrefixSumConfig(), InverseTransformSamplingConfig(512, 32)),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = ITSConfig(DecoupledPrefixSumConfig(), InverseTransformSamplingConfig(512, 128)),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },


    TestCase{
        .config = CutpointConfig(DecoupledPrefixSumConfig(), 32),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = CutpointConfig(DecoupledPrefixSumConfig(), 32),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = CutpointConfig(DecoupledPrefixSumConfig(), 32),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = CutpointConfig(DecoupledPrefixSumConfig(), 128),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },


    TestCase{
        .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                             DecoupledPrefixPartitionConfig(),
                                             InlineSplitPackConfig(2, 32, 512),
                                             false),
                                   SampleAliasTableConfig(32)),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                             DecoupledPrefixPartitionConfig(),
                                             InlineSplitPackConfig(2, 32, 512),
                                             false),
                                   SampleAliasTableConfig(32)),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                             DecoupledPrefixPartitionConfig(),
                                             InlineSplitPackConfig(2, 32, 512),
                                             false),
                                   SampleAliasTableConfig(128)),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                             DecoupledPrefixPartitionConfig(),
                                             InlineSplitPackConfig(2, 32, 512),
                                             false),
                                   SampleAliasTableConfig(128)),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                             DecoupledPrefixPartitionConfig(),
                                             InlineSplitPackConfig(2, 32, 512),
                                             false),
                                   SampleAliasTableConfig(0)),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
    TestCase{
        .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                             DecoupledPrefixPartitionConfig(),
                                             InlineSplitPackConfig(2, 32, 512),
                                             false),
                                   SampleAliasTableConfig(0)),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = static_cast<uint32_t>(1e8),
        .iterations = 1,
    },
};

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> weights) {
    Buffers::WeightsView stageView{stage.weights, weights.size()};
    Buffers::WeightsView localView{buffers.weights, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void downloadToStage(const merian::CommandBufferHandle& cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            ChiSquare::Buffers chiBuffers,
                            ChiSquare::Buffers chiStage,
                            host::glsl::uint S) {
    {
        Buffers::SamplesView stageView{stage.samples, S};
        Buffers::SamplesView localView{buffers.samples, S};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        ChiSquare::Buffers::ChiSquareView stageView{chiStage.chiSquare};
        ChiSquare::Buffers::ChiSquareView localView{chiBuffers.chiSquare};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results {
    std::pmr::vector<host::glsl::uint> samples;
    float chiSquare;
};
static Results downloadFromStage(Buffers& stage,
                                 ChiSquare::Buffers& chiStage,
                                 host::glsl::uint S,
                                 std::pmr::memory_resource* resource) {
    Buffers::SamplesView samplesView{stage.samples, S};
    auto samples =
        samplesView.download<host::glsl::uint, host::pmr_alloc<host::glsl::uint>>(resource);

    ChiSquare::Buffers::ChiSquareView chiView{chiStage.chiSquare};
    auto chiSquare = chiView.download<float>();

    return Results{
        .samples = std::move(samples),
        .chiSquare = chiSquare,
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
    Buffers buffers = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, testCase.N,
                                        testCase.S, testCase.config);
    Buffers stage = Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                      testCase.N, testCase.S, testCase.config);

    ChiSquare::Buffers chiBuffers =
        ChiSquare::Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, testCase.N,
                                     testCase.S, ChiSquareAllocFlags::ALLOC_CHI_SQUARE);
    chiBuffers.weights = buffers.weights;
    chiBuffers.samples = buffers.samples;

    ChiSquare::Buffers chiStage =
        ChiSquare::Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                     testCase.N, testCase.S, ChiSquareAllocFlags::ALLOC_CHI_SQUARE);
    chiStage.weights = stage.weights;
    chiStage.samples = stage.samples;

    std::string testName =
        fmt::format("{{{},N={},S={}}}", wrsConfigName(testCase.config), testCase.N, testCase.S);
    SPDLOG_INFO("Running test case:{}", testName);
    std::string recordingLabel = fmt::format("Recording : {}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    host::test::TestResultType out = host::test::SUCCESS;
    float averageJSDivergence = 0;
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
        auto weights =
            host::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
        /* std::ranges::sort(weights); */
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
        {
            /* MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, wrsConfigName(testCase.config)); */
            // 4. Run test case
            {
                MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Building WRS");
                SPDLOG_DEBUG("Building WRS");
                kernel.build(cmd, buffers, testCase.N, profiler);
            }

            { // 5. Samples WRS
                MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Sample WRS");
                SPDLOG_DEBUG("Sample WRS");
                std::random_device rng;
                std::uniform_int_distribution<host::glsl::uint> dist{};
                kernel.sample(cmd, buffers, testCase.N, testCase.S, dist(rng));
            }
        }

        // Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, chiBuffers, chiStage, testCase.S);
        }

        // Submit to device
        profiler->end();
        profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd->end();
        context.queue->submit_wait(cmd);

        // Download from stage
        profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results results = downloadFromStage(stage, chiStage, testCase.S, resource);
        profiler->end();

        // Test results
        {
            MERIAN_PROFILE_SCOPE(profiler, "Testing results");

            /* SPDLOG_DEBUG("Testing results"); */
            float jsDivergence =
                host::js_divergence<host::glsl::uint, host::glsl::f32>(results.samples, weights);

            averageJSDivergence += jsDivergence;
        }
        profiler->collect(true, true);
    }

    averageJSDivergence /= testCase.iterations;
    SPDLOG_INFO("JS-Divergence: {}", averageJSDivergence);

    if (averageJSDivergence > 0.3) {
        SPDLOG_ERROR("{} - WTF are you doing", wrsConfigName(testCase.config));
        out += host::test::ERROR;
    } else if (averageJSDivergence > 0.15) {
        SPDLOG_ERROR("{} displays a significant bias", wrsConfigName(testCase.config));
        out += host::test::ERROR;
    } else if (averageJSDivergence > 0.05) {
        SPDLOG_WARN("{} displays a moderate bias", wrsConfigName(testCase.config));
        out += host::test::WARNING;
    } else {
        SPDLOG_INFO("{} is does not show any significant bias", wrsConfigName(testCase.config));
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
    auto buildEntry = std::ranges::find_if(
        recordingEntry->children,
        [&](const merian::Profiler::ReportEntry& entry) { return entry.name == "Building WRS"; });
    auto sampleEntry = std::ranges::find_if(
        recordingEntry->children,
        [&](const merian::Profiler::ReportEntry& entry) { return entry.name == "Sample WRS"; });

    context.pushResult("WRS-Construction+Sampling", wrsConfigClass(testCase.config), out,
                       buildEntry->duration + sampleEntry->duration,
                       buildEntry->std_deviation + sampleEntry->std_deviation,
                       {
                           host::test::TestProperty{
                               .name = "N",
                               .value = fmt::format("{:.0e}", static_cast<float>(testCase.N)),
                           },
                           host::test::TestProperty{
                               .name = "S",
                               .value = fmt::format("{:.0e}", static_cast<float>(testCase.S)),
                           },
                           host::test::TestProperty{
                               .name = "section-size",
                               .value = fmt::format("{}", wrsConfigSectionSize(testCase.config)),
                           },
                       }

    );
}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing WRS algorithms");

    std::pmr::memory_resource* resource = context.memory_resource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, testCase, resource);
    }
}

} // namespace device::wrs
