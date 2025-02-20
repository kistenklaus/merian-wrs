#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/device/statistics/chi_square/ChiSquare.hpp"
#include "src/device/statistics/chi_square/ChiSquareAllocFlags.hpp"
#include "src/host/assert/test.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/memory/FallbackResource.hpp"
#include "src/host/memory/SafeResource.hpp"
#include "src/host/memory/StackResource.hpp"
#include "src/host/statistics/chi_square.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <random>
#include <spdlog/spdlog.h>

#include "src/device/wrs/WRS.hpp"
#include "src/host/reference/reduce.hpp"
#include "src/host/statistics/js_divergence.hpp"
#include "src/host/statistics/kl_divergence.hpp"
#include "src/host/why.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device::test::wrs {

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

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 0)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 4096, false)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 4096, true)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 1024, false)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 1024, true)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 512, false)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 512, true)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },

    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 128, false)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 128, true)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },


    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 32, false)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
    },
    TestCase{
        .config = ITSConfig( //
            DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
            InverseTransformSamplingConfig(512, 32, true)),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .S = 1024 * 2048 * 32,
        .iterations = 5,
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

static bool runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        const ChiSquare& chiSquare,
                        std::pmr::memory_resource* resource) {
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

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    bool failed = false;
    float averageJSDivergence = 0;
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
        const auto weights =
            host::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
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
            uploadTestCase(cmd, buffers, stage, weights);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, wrsConfigName(testCase.config));
            // 4. Run test case
            {
                MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Building WRS");
                SPDLOG_DEBUG("Building WRS");
                kernel.build(cmd, buffers, testCase.N);
            }

            { // 5. Samples WRS
                MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Sample WRS");
                SPDLOG_DEBUG("Sample WRS");
                std::random_device rng;
                std::uniform_int_distribution<host::glsl::uint> dist{};
                kernel.sample(cmd, buffers, testCase.N, testCase.S, dist(rng));
            }
        }

        const float totalWeight = host::reference::reduce<float>(weights);
        { // Chi Square (FUCKING USELESS SHIT)
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "ChiSquared (X²)");
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eComputeShader,
                         buffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                         vk::AccessFlagBits::eShaderRead));
            chiSquare.run(cmd, chiBuffers, testCase.N, testCase.S, totalWeight);
        }

        // Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, chiBuffers, chiStage, testCase.S);
        }

        // Submit to device
        context.profiler->end();
        context.profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd->end();
        context.queue->submit_wait(cmd);

        // Download from stage
        context.profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        [[maybe_unused]] Results results = downloadFromStage(stage, chiStage, testCase.S, resource);
        context.profiler->end();

        // Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            /* SPDLOG_DEBUG("Testing results"); */
            float jsDivergence =
                host::js_divergence<host::glsl::uint, host::glsl::f32>(results.samples, weights);

            averageJSDivergence += jsDivergence;
        }
        context.profiler->collect(true, true);
    }

    averageJSDivergence /= testCase.iterations;
    SPDLOG_INFO("JS-Divergence: {}", averageJSDivergence);

    if (averageJSDivergence > 0.3) {
        SPDLOG_ERROR("{} - WTF are you doing", wrsConfigName(testCase.config));
    } else if (averageJSDivergence > 0.15) {
        SPDLOG_ERROR("{} displays a significant bias", wrsConfigName(testCase.config));
    } else if (averageJSDivergence > 0.05) {
        SPDLOG_WARN("{} displays a moderate bias", wrsConfigName(testCase.config));
    } else {
        SPDLOG_INFO("{} is does not show any significant bias", wrsConfigName(testCase.config));
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing TODO algorithm");

    const host::test::TestContext testContext = host::test::setupTestContext(context);

    host::memory::StackResource stackResource{4096 * 2048};
    host::memory::FallbackResource fallbackResource{&stackResource};
    host::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    ChiSquare chiSquare{context, testContext.shaderCompiler};

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, chiSquare, resource);
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

} // namespace device::test::wrs
