#include "./wrs.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/WRS.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include "src/host/gen/weight_generator.h"
#include <csignal>
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <tuple>

namespace device::wrs {

struct NamedConfig {
    std::string name;
    WRS::Config config;
};

static const NamedConfig CONFIGURATIONS[] = {
    // NamedConfig{
    //    .name = "PSA-InlineSplitPack",
    //    .config = AliasTableConfig(               //*/
    //        PSAConfig(                            //*/
    //            AtomicMeanConfig(),               //*/
    //            DecoupledPrefixPartitionConfig(), //*/
    //            InlineSplitPackConfig(2),         //*/
    //            false),                           //*/
    //        SampleAliasTableConfig(32)),          //*/
    //},                                            //*/
    // NamedConfig{
    //    .name = "PSA-SerialSplitSubgroupPack",
    //    .config = AliasTableConfig(               //
    //        PSAConfig(                            //
    //            AtomicMeanConfig(),               //
    //            DecoupledPrefixPartitionConfig(), //
    //            SerialSplitPackConfig(ScalarSplitConfig(16),
    //                                  SubgroupPackConfig(16)), //
    //            false),                                        //
    //        SampleAliasTableConfig(32)),                       //
    //},                                                         //
    // NamedConfig{
    //    .name = "PSA-SerialSplitSubgroupPack",
    //    .config = AliasTableConfig(               //
    //        PSAConfig(                            //
    //            AtomicMeanConfig(),               //
    //            DecoupledPrefixPartitionConfig(), //
    //            SerialSplitPackConfig(ScalarSplitConfig(32),
    //                                  SubgroupPackConfig(32)), //
    //            false),                                        //
    //        SampleAliasTableConfig(32)),                       //
    //},                                                         //
    /* NamedConfig{ */
    /*     .name = "PSA-SerialSplitSubgroupPack-16-2/32", */
    /*     .config = AliasTableConfig(               // */
    /*         PSAConfig(                            // */
    /*             AtomicMeanConfig(),               // */
    /*             DecoupledPrefixPartitionConfig(), // */
    /*             SerialSplitPackConfig(ScalarSplitConfig(16), */
    /*                                   SubgroupPackConfig(16,2)), // */
    /*             true),                                        // */
    /*         SampleAliasTableConfig(32)),                       // */
    /* },                                                         // */
    /* NamedConfig{ */
    /*     .name = "PSA-SerialSplitSubgroupPack-32-2/32", */
    /*     .config = AliasTableConfig(               // */
    /*         PSAConfig(                            // */
    /*             AtomicMeanConfig(),               // */
    /*             DecoupledPrefixPartitionConfig(), // */
    /*             SerialSplitPackConfig(ScalarSplitConfig(32), */
    /*                                   SubgroupPackConfig(32, 2)), // */
    /*             true),                                        // */
    /*         SampleAliasTableConfig(32)),                       // */
    /* },                                                         // */
    NamedConfig{.name = "ITS-BINARY",
                .config = ITSConfig(DecoupledPrefixSumConfig(),
                                    InverseTransformSamplingConfig(512, 0, false))},
    NamedConfig{.name = "ITS-COOP",
                .config = ITSConfig(DecoupledPrefixSumConfig(),
                                    InverseTransformSamplingConfig(512, 128, false))},
    NamedConfig{.name = "ITS-COOP-PARRAY",
                .config = ITSConfig(DecoupledPrefixSumConfig(),
                                    InverseTransformSamplingConfig(512, 128, true))},
};

static constexpr std::size_t N = (1 << 26);
static constexpr std::size_t S = (1 << 28);

static constexpr std::size_t N_min = (1 << 12);
static constexpr std::size_t S_min = (1 << 21);

static constexpr std::size_t ticks = 25;
static constexpr std::size_t iterations = 1;

struct ConfigResult {
    std::size_t N;
    std::size_t S;
    double buildLatency;
    double buildStdVar;
    double samplingLatency;
    double samplingStdVar;
    double totalLatency;
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
                                       const std::size_t N,
                                       const std::size_t N_ticks,
                                       const std::size_t S,
                                       const std::size_t S_ticks,
                                       const std::size_t iterations,
                                       std::span<const float> weights) {
    SPDLOG_INFO("Benchmarking {}", wrsConfigName(config));
    assert(N > 1024);
    assert(S > 1024);
    assert(N_ticks > 2);
    assert(N_ticks > 2);
    assert(N <= (1 << 28));
    assert(S <= (1 << 28));

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    WRS wrs{context, shaderCompiler, config};

    WRS::Buffers local;
    { // Setup
        const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
        assert(resourceExt != nullptr);
        auto alloc = resourceExt->resource_allocator();

        WRS::Buffers stage = WRS::Buffers::allocate(
            alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, N, S, config);

        local = WRS::Buffers::allocate(alloc, merian::MemoryMappingType::NONE, N, S, config);

        WRS::Buffers::WeightsView weightsStageView{stage.weights, N};
        WRS::Buffers::WeightsView weightsLocalView{local.weights, N};

        weightsStageView.upload<float>(weights);

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();
        weightsStageView.copyTo(cmd, weightsLocalView);
        weightsLocalView.expectComputeRead(cmd);
        cmd->end();
        queue->submit_wait(cmd);
    }

    ConfigBenchmark results;
    results.entries.reserve(N_ticks * S_ticks);

    for (const std::size_t n : host::exp::log10scale<std::size_t>(N_min, N, N_ticks)) {
        SPDLOG_INFO("N = {}", n);

        merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
        merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
            std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(
                context, 8 * 2 * (S_ticks * iterations + iterations));
        query_pool->reset();
        profiler->set_query_pool(query_pool);

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);

        cmd->begin();
        for (std::size_t i = 0; i < iterations; ++i) {
            profiler->start("Build");
            profiler->cmd_start(cmd, "Build");
            wrs.build(cmd, local, n);
            profiler->end();
            profiler->cmd_end(cmd);
        }

        for (const std::size_t s : host::exp::log10scale<std::size_t>(S_min, S, S_ticks)) {
            std::string label = fmt::format("{}", s);
            for (std::size_t i = 0; i < iterations; ++i) {
                profiler->start(label);
                profiler->cmd_start(cmd, label);
                wrs.sample(cmd, local, n, s);
                profiler->end();
                profiler->cmd_end(cmd);
            }

            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eComputeShader,
                         local.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                       vk::AccessFlagBits::eShaderRead));
        }

        cmd->end();
        queue->submit_wait(cmd);
        profiler->collect(true, true);

        /* fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report())); */

        double buildLatency;
        double buildStdVar;
        std::vector<std::tuple<std::size_t, double, double>> samplingReport;
        samplingReport.reserve(S_ticks);
        for (const merian::Profiler::ReportEntry& entry : profiler->get_report().gpu_report) {
            if (entry.name == "Build") {
                buildLatency = entry.duration;
                buildStdVar = entry.std_deviation;
            } else {
                std::size_t s;
                std::istringstream(entry.name) >> s;
                samplingReport.push_back(std::make_tuple(s, entry.duration, entry.std_deviation));
            }
        }

        for (const auto report : samplingReport) {
            results.entries.push_back(ConfigResult{
                .N = n,
                .S = std::get<0>(report),
                .buildLatency = buildLatency,
                .buildStdVar = buildStdVar,
                .samplingLatency = std::get<1>(report),
                .samplingStdVar = std::get<2>(report),
                .totalLatency = buildLatency + std::get<1>(report),
            });
        }
    }

    return results;
}

void benchmark(const merian::ContextHandle& context) {
    // Setup vulkan resources
    merian::QueueHandle queue = context->get_queue_GCT();

    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);

    auto weights = host::generate_weights<float>(host::Distribution::PSEUDO_RANDOM_UNIFORM, N);

    BenchmarkResults results;
    for (const auto& config : CONFIGURATIONS) {
        auto configBenchmark = benchmarkConfiguration(context, shaderCompiler, queue, config.config,
                                                      N, ticks, S, ticks, iterations, weights);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
    }

    // export

    std::string path = "wrs_benchmark.csv";
    host::exp::CSVWriter<8> csv({"N", "S", "method", "build_latency", "build_std_derivation",
                                 "sampling_latency", "sampling_std_derivation", "total_latency"},
                                path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.S, method, r2.buildLatency, r2.buildStdVar, r2.samplingLatency,
                        r2.samplingStdVar, r2.totalLatency);
        }
    }
}

} // namespace device::wrs
