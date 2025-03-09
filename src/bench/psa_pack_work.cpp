#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/mean/Mean.hpp"
#include "src/device/mean/MeanAllocFlags.hpp"
#include "src/device/mean/atomic/AtomicMean.hpp"
#include "src/device/prefix_partition/PrefixPartition.hpp"
#include "src/device/prefix_partition/PrefixPartitionAllocFlags.hpp"
#include "src/device/prng/PRNG.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include "src/device/wrs/alias/psa/pack/Pack.hpp"
#include "src/device/wrs/alias/psa/pack/PackAllocFlags.hpp"
#include "src/device/wrs/alias/psa/pack/scalar/ScalarPack.hpp"
#include "src/device/wrs/alias/psa/split/Split.hpp"
#include "src/device/wrs/alias/psa/split/SplitAllocFlags.hpp"
#include "src/device/wrs/alias/psa/split/scalar/ScalarSplit.hpp"
#include "src/host/export/csv.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <random>
#include <spdlog/spdlog.h>

namespace device::psa_pack_work {

using weight_type = float;
struct NamedConfig {
    std::string name;
    std::string group;
    PackConfig config;
    bool flushL2;
};

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{//
                .name = "ScalarPack-512-2",
                .group = "Pack1-2",
                .config = ScalarPackConfig(2),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-2-2",
                .group = "Pack2-2",
                .config = SubgroupPackConfig(2, 16),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-2-4",
                .group = "Pack2-4",
                .config = SubgroupPackConfig(4, 16),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-2-8",
                .group = "Pack2-8",
                .config = SubgroupPackConfig(8, 16),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-4-4",
                .group = "Pack4-4",
                .config = SubgroupPackConfig(4, 8),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-4-8",
                .group = "Pack4-8",
                .config = SubgroupPackConfig(8, 8),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-4-16",
                .group = "Pack4-16",
                .config = SubgroupPackConfig(16, 8),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-4-32",
                .group = "Pack4-32",
                .config = SubgroupPackConfig(32, 8),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-4-64",
                .group = "Pack4-64",
                .config = SubgroupPackConfig(64, 8),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-8-16",
                .group = "Pack8-16",
                .config = SubgroupPackConfig(16, 4),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-8-32",
                .group = "Pack8-32",
                .config = SubgroupPackConfig(32, 4),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-8-64",
                .group = "Pack8-64",
                .config = SubgroupPackConfig(64, 4),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-8-128",
                .group = "Pack8-128",
                .config = SubgroupPackConfig(128, 4),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-16-32",
                .group = "Pack16-32",
                .config = SubgroupPackConfig(32, 2),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-16-64",
                .group = "Pack16-64",
                .config = SubgroupPackConfig(64, 2),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-16-128",
                .group = "Pack16-128",
                .config = SubgroupPackConfig(128, 2),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-16-256",
                .group = "Pack16-256",
                .config = SubgroupPackConfig(256, 2),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-32-32",
                .group = "Pack32-32",
                .config = SubgroupPackConfig(32, 1),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-32-64",
                .group = "Pack32-64",
                .config = SubgroupPackConfig(64, 1),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-32-128",
                .group = "Pack32-128",
                .config = SubgroupPackConfig(128, 1),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-32-256",
                .group = "Pack32-256",
                .config = SubgroupPackConfig(256, 1),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SubgroupPack-512-32-512",
                .group = "Pack32-512",
                .config = SubgroupPackConfig(512, 1),
                .flushL2 = true}, //

    /* // */
    /* NamedConfig{// */
    /*             .name = "ScalarPack-512-2", */
    /*             .group = "Pack1-2", */
    /*             .config = ScalarPackConfig(2), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-2-2", */
    /*             .group = "Pack2-2", */
    /*             .config = SubgroupPackConfig(2, 16), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-2-4", */
    /*             .group = "Pack2-4", */
    /*             .config = SubgroupPackConfig(4, 16), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-2-8", */
    /*             .group = "Pack2-8", */
    /*             .config = SubgroupPackConfig(8, 16), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-4-4", */
    /*             .group = "Pack4-4", */
    /*             .config = SubgroupPackConfig(4, 8), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-4-8", */
    /*             .group = "Pack4-8", */
    /*             .config = SubgroupPackConfig(8, 8), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-4-16", */
    /*             .group = "Pack4-16", */
    /*             .config = SubgroupPackConfig(16, 8), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-4-32", */
    /*             .group = "Pack4-32", */
    /*             .config = SubgroupPackConfig(32, 8), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-4-64", */
    /*             .group = "Pack4-64", */
    /*             .config = SubgroupPackConfig(64, 8), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-8-16", */
    /*             .group = "Pack8-16", */
    /*             .config = SubgroupPackConfig(16, 4), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-8-32", */
    /*             .group = "Pack8-32", */
    /*             .config = SubgroupPackConfig(32, 4), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-8-64", */
    /*             .group = "Pack8-64", */
    /*             .config = SubgroupPackConfig(64, 4), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-8-128", */
    /*             .group = "Pack8-128", */
    /*             .config = SubgroupPackConfig(128, 4), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-16-32", */
    /*             .group = "Pack16-32", */
    /*             .config = SubgroupPackConfig(32, 2), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-16-64", */
    /*             .group = "Pack16-64", */
    /*             .config = SubgroupPackConfig(64, 2), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-16-128", */
    /*             .group = "Pack16-128", */
    /*             .config = SubgroupPackConfig(128, 2), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-16-256", */
    /*             .group = "Pack16-256", */
    /*             .config = SubgroupPackConfig(256, 2), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-32-32", */
    /*             .group = "Pack32-32", */
    /*             .config = SubgroupPackConfig(32, 1), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-32-64", */
    /*             .group = "Pack32-64", */
    /*             .config = SubgroupPackConfig(64, 1), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-32-128", */
    /*             .group = "Pack32-128", */
    /*             .config = SubgroupPackConfig(128, 1), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-32-256", */
    /*             .group = "Pack32-256", */
    /*             .config = SubgroupPackConfig(256, 1), */
    /*             .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*             .name = "SubgroupPack-512-32-512", */
    /*             .group = "Pack32-512", */
    /*             .config = SubgroupPackConfig(512, 1), */
    /*             .flushL2 = false}, // */

};

static constexpr float w_min = 0.01; // min work coefficient
static constexpr float w_max = 1024;  // max work coefficient

// RTX 4070 with 46 SMs, 48 warps per SM and 32 threads per warp.
static constexpr std::size_t maxOccupantInvoc = 70656;

static constexpr std::size_t ticks = 100;

static constexpr std::size_t iterations = 1;
static constexpr std::size_t flushSize = 1e8;

static constexpr MeanConfig meanConfig = AtomicMeanConfig();
static constexpr PrefixPartitionConfig prefixPartitionConfig = DecoupledPrefixPartitionConfig();

struct ConfigResult {
    std::size_t N;
    std::size_t splitSize;
    std::size_t invocPerPack;
    float w;
    double latency; // ms
    double stdVar;  // ms
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
                                       PackConfig config,
                                       bool flushL2) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
    assert(resourceExt != nullptr);
    auto alloc = resourceExt->resource_allocator();

    std::size_t splitSize = packConfigSplitSize(config);
    std::size_t invocPerPack = packConfigInvocPerPack(context, config);
    std::size_t maxThreads = std::ceil(w_max * maxOccupantInvoc);
    std::size_t maxK = (maxThreads + invocPerPack - 1) / invocPerPack; // max packs
    std::size_t maxN = std::min<std::size_t>(maxK * splitSize, (1 << 28));
    maxK = (maxN + splitSize - 1) / splitSize;

    PhiloxBuffers weights = PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, maxN);
    PhiloxBuffers flush =
        PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, flushSize);

    MeanBuffers meanBuffers =
        MeanBuffers::allocate<weight_type>(alloc, merian::MemoryMappingType::NONE, meanConfig, maxN,
                                           MeanAllocFlags::ALLOC_ONLY_OUTPUT);
    meanBuffers.elements = weights.samples;
    PrefixPartitionBuffers prefixPartitionBuffers = PrefixPartitionBuffers::allocate<weight_type>(
        alloc, merian::MemoryMappingType::NONE, prefixPartitionConfig, maxN,
        PrefixPartitionAllocFlags::ALLOC_ONLY_OUTPUT);
    prefixPartitionBuffers.elements = weights.samples;
    prefixPartitionBuffers.pivot = meanBuffers.mean;

    SplitBuffers splitBuffers = SplitBuffers::allocate(alloc, merian::MemoryMappingType::NONE, maxN,
                                                       maxK, SplitAllocFlags::ALLOC_ONLY_OUTPUTS);
    splitBuffers.partitionPrefix = prefixPartitionBuffers.partitionPrefix;
    splitBuffers.mean = meanBuffers.mean;
    splitBuffers.heavyCount = prefixPartitionBuffers.heavyCount;

    PackBuffers packBuffers = PackBuffers::allocate(alloc, merian::MemoryMappingType::NONE, maxN,
                                                    maxK, PackAllocFlags::ALLOC_ALL_OUTPUTS);
    packBuffers.heavyCount = prefixPartitionBuffers.heavyCount;
    packBuffers.mean = meanBuffers.mean;
    packBuffers.partitionElements = prefixPartitionBuffers.partitionElements;
    packBuffers.partitionIndices = prefixPartitionBuffers.partitionIndices;
    packBuffers.splits = splitBuffers.splits;
    packBuffers.weights = weights.samples;

    Mean<weight_type> mean{context, shaderCompiler, meanConfig};
    PrefixPartition<weight_type> prefixPartition{context, shaderCompiler, prefixPartitionConfig,
                                                 false};

    Split split{context, shaderCompiler, ScalarSplitConfig(packConfigSplitSize(config))};

    Pack pack{context, shaderCompiler, config, false};

    ConfigBenchmark results;

    PRNG prng{context, shaderCompiler, PhiloxConfig()};
    PRNGBuffers prngBuffers;
    prngBuffers.samples = weights.samples;
    PRNGBuffers flushBuffers;
    flushBuffers.samples = flush.samples;

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;

    float stepW = (w_max - w_min) / ticks;
    for (float w = w_min; w <= w_max; w += stepW) {

        std::size_t invocations = std::ceil(w * maxOccupantInvoc);
        std::size_t K = (invocations + invocPerPack - 1) / invocPerPack;
        std::size_t n = K * splitSize;

        if (K > maxK) {
          fmt::println("Skip remaining ticks, K is to large");
          break;
        }

        if (n > maxN) {
          fmt::println("Skip remaining ticks, N is to large");
          break;
        }

        merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
        merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
            std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 4 * iterations);
        query_pool->reset();
        profiler->set_query_pool(query_pool);

        for (std::size_t i = 0; i < iterations; ++i) {
            {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();
                prng.run(cmd, prngBuffers, n, dist(rng));
                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                mean.run(cmd, meanBuffers, n);

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             meanBuffers.mean->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead));

                prefixPartition.run(cmd, prefixPartitionBuffers, n);
                cmd->barrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {
                        prefixPartitionBuffers.partitionPrefix->buffer_barrier(
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                        prefixPartitionBuffers.partitionIndices->buffer_barrier(
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                        prefixPartitionBuffers.heavyCount->buffer_barrier(
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                    });

                split.run(cmd, splitBuffers, n);

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             splitBuffers.splits->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                if (flushL2) {
                    /* prng.run(cmd, flushBuffers, flushSize, dist(rng)); */
                    /* cmd->barrier( */
                    /*     vk::PipelineStageFlagBits::eComputeShader, */
                    /*     vk::PipelineStageFlagBits::eComputeShader, */
                    /*     flushBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite, */
                    /*                                          vk::AccessFlagBits::eShaderRead)); */
                }

                profiler->start("Pack");
                profiler->cmd_start(cmd, "Pack");

                pack.run(cmd, packBuffers, n);

                profiler->end();
                profiler->cmd_end(cmd);

                cmd->end();
                queue->submit_wait(cmd);
            }
            profiler->collect(true, true);
        }

        auto report = profiler->get_report();
        auto entry = std::ranges::find_if(report.gpu_report,
                                          [](const auto& entry) { return entry.name == "Pack"; });
        assert(entry != report.gpu_report.end());
        double latency = entry->duration;
        double stdVar = entry->std_deviation;

        results.entries.push_back(ConfigResult{
            .N = n,
            .splitSize = splitSize,
            .invocPerPack = invocPerPack,
            .w = w,
            .latency = latency,
            .stdVar = stdVar,
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
    SPDLOG_INFO("Benchmarking PSA-Pack (latency relative to the work coefficient)");
    for (const auto& config : CONFIGURATIONS) {
        SPDLOG_INFO(
            "[{}%] Benchmarking {} ({})",
            (i / static_cast<float>(sizeof(CONFIGURATIONS) / (float)sizeof(CONFIGURATIONS[0]))) *
                100.0f,
            config.group, config.name);
        auto configBenchmark =
            benchmarkConfiguration(context, shaderCompiler, queue, config.config, config.flushL2);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
        ++i;
    }

    // export

    std::string path = "psa_pack_benchmark_latency.csv";
    host::exp::CSVWriter<9> csv({"N", "splitSize", "w", "invocPerPack", "method", "group",
                                 "latency", "std_derivation", "flushL2"},
                                path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.splitSize, r2.w, r2.invocPerPack, method, r1.configuration.group,
                        r2.latency, r2.stdVar, r1.configuration.flushL2);
        }
    }
}

} // namespace device::psa_pack_work
