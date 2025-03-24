#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/PrefixSum.hpp"
#include "src/device/prng/PRNG.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include "src/device/wrs/WRS.hpp"
#include "src/device/wrs/its/ITS.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/statistics/rmse.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <csignal>
#include <fmt/base.h>
#include <random>
#include <spdlog/spdlog.h>

namespace device::wrs_rmse_sweep {

using weight_type = float;
using Buffers = WRS::Buffers;
struct NamedConfig {
    std::string name;
    std::string group;
    WRS::Config config;
};

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{.name = "ITS-0",
                .group = "ITS-0",
                .config = ITSConfig(DecoupledPrefixSumConfig(),
                                    InverseTransformSamplingConfig(128, 0, false))},
    NamedConfig{.name = "ITS-128",
                .group = "ITS-128",
                .config = ITSConfig(DecoupledPrefixSumConfig(),
                                    InverseTransformSamplingConfig(128, 128, false))},

    NamedConfig{.name = "Cutpoint-128",
                .group = "Cutpoint-128",
                .config = CutpointConfig(DecoupledPrefixSumConfig(), 128)},

    NamedConfig{.name = "PSA2-0",
                .group = "PSA2-0",
                .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                                     DecoupledPrefixPartitionConfig(),
                                                     InlineSplitPackConfig(2, 32),
                                                     false),
                                           SampleAliasTableConfig(0))},
    NamedConfig{.name = "PSA2-128",
                .group = "PSA2-128",
                .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                                     DecoupledPrefixPartitionConfig(),
                                                     InlineSplitPackConfig(2, 32),
                                                     false),
                                           SampleAliasTableConfig(128))},

};

static constexpr std::size_t min_N = (1 << 16);
static constexpr std::size_t max_N = (1 << 26);
static constexpr auto weight_distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM;
static constexpr std::size_t min_S = (1 << 16);
static constexpr std::size_t max_S = (1 << 28);
static constexpr std::size_t ticks = 1000;
static constexpr std::size_t flushSize = 1e7;

struct ConfigResult {
    std::size_t N;
    std::size_t S;
    double rmse;
};

struct ConfigBenchmark {
    std::vector<ConfigResult> entries;
};

struct BenchmarkResult {
    NamedConfig configuration;
    ConfigBenchmark results;
};

struct BenchmarkResults {
    std::vector<BenchmarkResult> entries;
};

ConfigBenchmark benchmarkConfiguration(const merian::ContextHandle& context,
                                       const merian::ShaderCompilerHandle& shaderCompiler,
                                       const merian::QueueHandle& queue,
                                       const WRS::Config& config,
                                       host::glsl::uint N) {

    constexpr host::glsl::uint MAX_SAMPLING_STEP_SIZE = (1 << 28);
    constexpr host::glsl::uint SAMPLING_STEP_COUNT =
        (max_S + static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE) - 1) /
        static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE);
    constexpr host::glsl::uint SUBMIT_LIMIT = 4;

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    WRS wrs{context, shaderCompiler, config};

    auto weights = host::generate_weights<float>(weight_distribution, N);
    auto totalWeight = host::reference::reduce<float>(weights);

    Buffers local;
    PhiloxBuffers temp;
    { // Setup
        const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
        assert(resourceExt != nullptr);
        auto alloc = resourceExt->resource_allocator();

        local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, N, MAX_SAMPLING_STEP_SIZE,
                                  config);
        Buffers stage = Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, N,
                                          max_S, config);

        temp = PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, flushSize);

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();

        Buffers::WeightsView stageView{stage.weights, N};
        Buffers::WeightsView localView{local.weights, N};
        stageView.upload<float>(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);

        wrs.build(cmd, local, N);

        cmd->end();
        queue->submit_wait(cmd);
    }

    wrs::eval::RMSECurveAcceleratedBuilder rmseCurveBuilder{
        context,
        shaderCompiler,
        local.weights,
        totalWeight,
        N,
        host::exp::log10scale<uint64_t>(min_S, max_S, ticks)};

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;
    std::size_t s = max_S;

    for (std::size_t i = 0; i < SAMPLING_STEP_COUNT;) {

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();

        std::size_t x = 0;
        while (i < SAMPLING_STEP_COUNT && x < SUBMIT_LIMIT) {

            std::size_t s2 = s;
            if (s2 == 0) {
              i = SAMPLING_STEP_COUNT;
              break;
            }
            if (s2 > MAX_SAMPLING_STEP_SIZE) {
                s2 = MAX_SAMPLING_STEP_SIZE;
            }
            s -= MAX_SAMPLING_STEP_SIZE;

            wrs.sample(cmd, local, N, s2, dist(rng));

            rmseCurveBuilder.consume(cmd, local.samples, s2);
            ++i;
            ++x;
        }

        /* SPDLOG_INFO("Sectioned Sampling: {}/{} ~ {:.3}%", max_S - s, max_S, */
        /*             100 * ((max_S - s) / static_cast<float>(max_S))); */

        cmd->end();
        queue->submit_wait(cmd);
    }

    std::span<const std::tuple<uint64_t, float>> rmseCurve = rmseCurveBuilder.get();

    ConfigBenchmark results;
    results.entries.reserve(rmseCurve.size());

    for (const auto& [s, rmse] : rmseCurve) {
        results.entries.push_back(ConfigResult{
            .N = N,
            .S = s,
            .rmse = rmse,
        });
    }

    return results;
}

void benchmark(const merian::ContextHandle& context) {
    // Setup vulkan resources
    merian::QueueHandle queue = context->get_queue_GCT();

    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);
    SPDLOG_INFO("Running RMSE sweep for WRS algorithms");

    BenchmarkResults results;
    std::size_t i = 0;

    std::string path = "wrs_rmse_sweep.csv";
    host::exp::CSVWriter<5> csv({"N", "S", "method", "group", "rmse"}, path);
    for (const auto& config : CONFIGURATIONS) {
        for (const auto n : host::exp::log10scale(min_N, max_N, ticks)) {
            SPDLOG_INFO("[{}%] Benchmarking {}",
                        ((i / static_cast<float>(sizeof(CONFIGURATIONS) /
                                                (float)sizeof(CONFIGURATIONS[0]))) / ticks) *
                            100.0f,
                        config.name);
            auto configBenchmark =
                benchmarkConfiguration(context, shaderCompiler, queue, config.config, n);
            for (const auto& r1 : configBenchmark.entries) {
              csv.pushRow(r1.N, r1.S, config.name, config.name, r1.rmse);
            }
            ++i;
        }
    }
}

} // namespace device::wrs_rmse_sweep
