#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/PrefixSum.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/device/prng/PRNG.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include "src/device/wrs/WRS.hpp"
#include "src/device/wrs/its/ITS.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <random>
#include <spdlog/spdlog.h>

namespace device::sample_throughput {

using weight_type = float;
using Buffers = WRS::Buffers;
struct NamedConfig {
    std::string name;
    std::string group;
    WRS::Config config;
    bool flushL2;
};

static const NamedConfig CONFIGURATIONS[] = {
    //NamedConfig{.name = "ITS-Scan-512-8-STRIDE)-(Sample-Binary-512)",
    //            .group = "ITS-Binary",
    //            .config =
    //                ITSConfig(DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
    //                          InverseTransformSamplingConfig(512, 0, false)),
    //            .flushL2 = true},
    //NamedConfig{.name = "ITS-Scan-512-8-STRIDE)-(Sample-Binary-512)",
    //            .group = "ITS-Binary",
    //            .config =
    //                ITSConfig(DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
    //                          InverseTransformSamplingConfig(512, 0, false)),
    //            .flushL2 = false},
    //NamedConfig{.name = "ITS-Scan-512-8-STRIDE)-(Sample-BinaryCoop-512-128)",
    //            .group = "ITS-BinaryCoop",
    //            .config =
    //                ITSConfig(DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
    //                          InverseTransformSamplingConfig(512, 128, false)),
    //            .flushL2 = true},
    //NamedConfig{.name = "ITS-Scan-512-8-STRIDE)-(Sample-BinaryCoop-512-128)",
    //            .group = "ITS-BinaryCoop",
    //            .config =
    //                ITSConfig(DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
    //                          InverseTransformSamplingConfig(512, 128, false)),
    //            .flushL2 = false},
    //NamedConfig{.name = "ITS-Scan-512-8-STRIDE)-(Sample-pArrayCoop-512-128)",
    //            .group = "ITS-pArrayCoop",
    //            .config =
    //                ITSConfig(DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
    //                          InverseTransformSamplingConfig(512, 128, true)),
    //            .flushL2 = true},
    //NamedConfig{.name = "ITS-Scan-512-8-STRIDE)-(Sample-pArrayCoop-512-128)",
    //            .group = "ITS-pArrayCoop",
    //            .config =
    //                ITSConfig(DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
    //                          InverseTransformSamplingConfig(512, 128, true)),
    //            .flushL2 = false},
    //NamedConfig{.name = "Cutpoint-128",
    //            .group = "Cutpoint",
    //            .config = CutpointConfig(
    //                DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED), 128),
    //            .flushL2 = true},
    //NamedConfig{.name = "Cutpoint-128",
    //            .group = "Cutpoint",
    //            .config = CutpointConfig(
    //                DecoupledPrefixSumConfig(512, 8, BlockScanVariant::RANKED_STRIDED), 128),
    //            .flushL2 = false},
    NamedConfig{.name = "PSA-(Inline-2-no-elements)",
                .group = "PSA-Inline",
                .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                                     DecoupledPrefixPartitionConfig(),
                                                     InlineSplitPackConfig(2),
                                                     false),
                                           SampleAliasTableConfig(128)),
                .flushL2 = true},
    NamedConfig{.name = "PSA-(Inline-2-no-elements)",
                .group = "PSA-Inline",
                .config = AliasTableConfig(PSAConfig(AtomicMeanConfig(),
                                                     DecoupledPrefixPartitionConfig(),
                                                     InlineSplitPackConfig(2),
                                                     false),
                                           SampleAliasTableConfig(128)),
                .flushL2 = false},

};

static constexpr std::size_t N = (1 << 28);
static constexpr std::size_t N_min = (1 << 16);
static constexpr std::size_t ticks = 1000;
static constexpr std::size_t iterations = 100;
static constexpr std::size_t S = 1e7;
static constexpr std::size_t flushSize = 1e7;

struct ConfigResult {
    std::size_t N;
    std::size_t S;
    double latencyBuild; // ms
    double stdVarBuild;  // ms
    double latencySample;
    double stdVarSample;
    double sampleThroughput; // billions / second
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
                                       bool flushL2) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    WRS wrs{context, shaderCompiler, config};

    Buffers local;
    PhiloxBuffers temp;
    { // Setup
        const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
        assert(resourceExt != nullptr);
        auto alloc = resourceExt->resource_allocator();

        local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, N, S, config);

        temp = PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, flushSize);
    }

    ConfigBenchmark results;
    results.entries.reserve(ticks);

    PRNG prng{context, shaderCompiler, PhiloxConfig(512)};
    PRNGBuffers prngBuffers;
    prngBuffers.samples = local.weights;
    PRNGBuffers flushBuffers;
    flushBuffers.samples = temp.samples;

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;

    for (const std::size_t n : host::exp::log10scale<std::size_t>(N_min, N, ticks)) {
        merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
        merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
            std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 4 * iterations);
        query_pool->reset();
        profiler->set_query_pool(query_pool);

        if (flushL2) {
            { // Generate random input
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();

                prng.run(cmd, prngBuffers, n, dist(rng));

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));
                // flush weights
                prng.run(cmd, flushBuffers, flushSize);
                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                cmd->end();
                queue->submit_wait(cmd);
            }

            {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();
                profiler->start("Build");
                profiler->cmd_start(cmd, "Build");
                wrs.build(cmd, local, n);
                profiler->end();
                profiler->cmd_end(cmd);

                // flush wrs data structure
                prng.run(cmd, flushBuffers, flushSize);
                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                cmd->end();
                queue->submit_wait(cmd);
            }
            for (std::size_t i = 0; i < iterations; ++i) {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();
                profiler->start("Sample");
                profiler->cmd_start(cmd, "Sample");
                wrs.sample(cmd, local, n, S, dist(rng));
                profiler->end();
                profiler->cmd_end(cmd);

                // flush writes and wrs data structure
                prng.run(cmd, flushBuffers, flushSize);
                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                cmd->end();
                queue->submit_wait(cmd);
                profiler->collect(true, true);
            }
        } else {
            for (std::size_t i = 0; i < iterations; ++i) {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();
                { // Generate weights
                    prng.run(cmd, prngBuffers, N);
                    cmd->barrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eShaderRead));
                }
                { // Build
                    profiler->start("Build");
                    profiler->cmd_start(cmd, "Build");
                    wrs.build(cmd, local, n);
                    profiler->end();
                    profiler->cmd_end(cmd);
                }
                { // Sample
                    profiler->start("Sample");
                    profiler->cmd_start(cmd, "Sample");
                    wrs.sample(cmd, local, n, S, dist(rng));
                    profiler->end();
                    profiler->cmd_end(cmd);
                }

                cmd->end();
                queue->submit_wait(cmd);
                profiler->collect(true, true);
            }
        }

        auto report = profiler->get_report();
        auto entryBuild = std::ranges::find_if(
            report.gpu_report, [](const auto& entry) { return entry.name == "Build"; });
        auto entrySample = std::ranges::find_if(
            report.gpu_report, [](const auto& entry) { return entry.name == "Sample"; });

        double latencyBuild = entryBuild->duration;
        double stdVarBuild = entryBuild->std_deviation;
        double latencySample = entrySample->duration;
        double stdVarSample = entrySample->std_deviation;

        double sampleThroughput = static_cast<double>(S) / (latencySample * 1e-3);

        results.entries.push_back(ConfigResult{
            .N = n,
            .S = S,
            .latencyBuild = latencyBuild,
            .stdVarBuild = stdVarBuild,
            .latencySample = latencySample,
            .stdVarSample = stdVarSample,
            .sampleThroughput = sampleThroughput,
        });
    }

    return results;
}

void benchmark(const merian::ContextHandle& context) {
    // Setup vulkan resources
    merian::QueueHandle queue = context->get_queue_GCT();

    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);

    BenchmarkResults results;
    std::size_t i = 0;
    for (const auto& config : CONFIGURATIONS) {
        SPDLOG_INFO(
            "[{}%] Benchmarking {}",
            (i / static_cast<float>(sizeof(CONFIGURATIONS) / (float)sizeof(CONFIGURATIONS[0]))) *
                100.0f,
            config.name);
        auto configBenchmark =
            benchmarkConfiguration(context, shaderCompiler, queue, config.config, config.flushL2);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
        ++i;
    }

    // export

    std::string path = "wrs_benchmark_sample_throughput.csv";
    host::exp::CSVWriter<10> csv({"N", "S", "method", "group", "build_latency",
                                  "build_std_derivation", "sample_latency", "sample_std_derivation",
                                  "sample_throughput", "flushL2"},
                                 path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.S, method, r1.configuration.group, r2.latencyBuild, r2.stdVarBuild,
                        r2.latencySample, r2.stdVarSample, r2.sampleThroughput,
                        r1.configuration.flushL2);
        }
    }
}

} // namespace device::sample_throughput
