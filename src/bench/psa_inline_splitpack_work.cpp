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
#include "src/device/wrs/alias/psa/PSA.hpp"
#include "src/device/wrs/alias/psa/pack/Pack.hpp"
#include "src/device/wrs/alias/psa/pack/PackAllocFlags.hpp"
#include "src/device/wrs/alias/psa/pack/scalar/ScalarPack.hpp"
#include "src/device/wrs/alias/psa/split/Split.hpp"
#include "src/device/wrs/alias/psa/split/SplitAllocFlags.hpp"
#include "src/device/wrs/alias/psa/split/scalar/ScalarSplit.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPack.hpp"
#include "src/host/export/csv.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <fmt/format.h>
#include <random>
#include <spdlog/spdlog.h>

namespace device::psa_inline_splitpack_work {

using weight_type = float;
struct NamedConfig {
    std::string name;
    std::string group;
    SplitPackConfig config;
    bool flushL2;
};

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{//
                .name = "SplitPack-1-2",
                .group = "SplitPack-1",
                .config = InlineSplitPackConfig(2, 32),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-2-4",
                .group = "SplitPack-2",
                .config = InlineSplitPackConfig(4, 16),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-4-12",
                .group = "SplitPack-4",
                .config = InlineSplitPackConfig(12, 8),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-8-24",
                .group = "SplitPack-8",
                .config = InlineSplitPackConfig(24, 4),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-16-54",
                .group = "SplitPack-16",
                .config = InlineSplitPackConfig(54, 2),
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-32-127",
                .group = "SplitPack-32",
                .config = InlineSplitPackConfig(127, 1),
                .flushL2 = true}, //
                                  //
    NamedConfig{                  //
                .name = "SplitPack-1-2",
                .group = "SplitPack-1",
                .config = InlineSplitPackConfig(2, 32),
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SplitPack-2-4",
                .group = "SplitPack-2",
                .config = InlineSplitPackConfig(4, 16),
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SplitPack-4-12",
                .group = "SplitPack-4",
                .config = InlineSplitPackConfig(12, 8),
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SplitPack-8-24",
                .group = "SplitPack-8",
                .config = InlineSplitPackConfig(24, 4),
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SplitPack-16-54",
                .group = "SplitPack-16",
                .config = InlineSplitPackConfig(54, 2),
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SplitPack-32-127",
                .group = "SplitPack-32",
                .config = InlineSplitPackConfig(127, 1),
                .flushL2 = false}, //
};

static constexpr float w_min = 0.01; // min work coefficient
static constexpr float w_max = 10;  // max work coefficient

// RTX 4070 with 46 SMs, 48 warps per SM and 32 threads per warp.
static constexpr std::size_t maxOccupantInvoc = 70656;

static constexpr std::size_t ticks = 1000;

static constexpr std::size_t iterations = 100;
static constexpr std::size_t flushSize = 1e8;

static constexpr MeanConfig meanConfig = AtomicMeanConfig();
static constexpr PrefixPartitionConfig prefixPartitionConfig = DecoupledPrefixPartitionConfig();

static constexpr bool writeElements = false;

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
                                       SplitPackConfig config,
                                       bool flushL2) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
    assert(resourceExt != nullptr);
    auto alloc = resourceExt->resource_allocator();

    std::size_t splitSize = splitPackConfigSplitSize(config);
    std::size_t invocPerPack = splitPackConfigInvocPerPack(context, config);
    std::size_t maxThreads = std::ceil(w_max * maxOccupantInvoc);
    std::size_t maxK = (maxThreads + invocPerPack - 1) / invocPerPack; // max packs
    std::size_t maxN = std::min<std::size_t>(maxK * splitSize, (1 << 28));
    maxK = (maxN + splitSize - 1) / splitSize;

    PhiloxBuffers flush =
        PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, flushSize);

    PSABuffers psaBuffers = PSABuffers::allocate(
        alloc, merian::MemoryMappingType::NONE,
        PSAConfig(meanConfig, prefixPartitionConfig, config, writeElements), maxN);
    PhiloxBuffers weights;
    weights.samples = psaBuffers.weights;

    MeanBuffers meanBuffers;
    meanBuffers.elements = psaBuffers.weights;
    meanBuffers.mean = psaBuffers.m_mean;
    meanBuffers.m_internalBuffers = psaBuffers.m_meanInternals;

    PrefixPartitionBuffers prefixPartitionBuffers;
    prefixPartitionBuffers.elements = psaBuffers.weights;
    prefixPartitionBuffers.pivot = psaBuffers.m_mean;
    prefixPartitionBuffers.partitionPrefix = psaBuffers.m_partitionPrefix;
    prefixPartitionBuffers.partitionElements = psaBuffers.m_partitionElements;
    prefixPartitionBuffers.partitionIndices = psaBuffers.m_partitionIndices;
    prefixPartitionBuffers.heavyCount = psaBuffers.m_heavyCount;
    prefixPartitionBuffers.m_internalBuffers = psaBuffers.m_prefixPartitionInternals;

    SplitPackBuffers splitPackBuffers = SplitPackBuffers::allocate(
        alloc, merian::MemoryMappingType::NONE, InlineSplitPackConfig(0, 0, 0), maxN,
        SplitPackAllocFlags::ALLOC_ONLY_INTERNALS);
    splitPackBuffers.heavyCount = psaBuffers.m_heavyCount;
    splitPackBuffers.mean = psaBuffers.m_mean;
    splitPackBuffers.partitionPrefix = psaBuffers.m_partitionPrefix;
    splitPackBuffers.partitionElements = psaBuffers.m_partitionElements;
    splitPackBuffers.partitionIndices = psaBuffers.m_partitionIndices;
    splitPackBuffers.weights = psaBuffers.weights;
    splitPackBuffers.aliasTable = psaBuffers.aliasTable;
    splitPackBuffers.m_internals = SplitPackBuffers::InlineInternals();

    Mean<weight_type> mean{context, shaderCompiler, meanConfig};
    PrefixPartition<weight_type> prefixPartition{context, shaderCompiler, prefixPartitionConfig,
                                                 false};

    SplitPack splitPack{context, shaderCompiler, config, writeElements};

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
        merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
        merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
            std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context,
                                                                           128 * iterations);
        query_pool->reset();
        profiler->set_query_pool(query_pool);

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

        std::string profilerLabel = fmt::format("splitpack-w={}", w);

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

                if (flushL2) {
                    prng.run(cmd, flushBuffers, flushSize);

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
                            flushBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead),
                        });
                }

                profiler->start(profilerLabel);
                profiler->cmd_start(cmd, profilerLabel);

                splitPack.run(cmd, splitPackBuffers, n);

                profiler->end();
                profiler->cmd_end(cmd);

                cmd->end();
                queue->submit_wait(cmd);
            }
            profiler->collect(true, true);
        }

        auto report = profiler->get_report();
        auto entry = std::ranges::find_if(
            report.gpu_report, [&](const auto& entry) { return entry.name == profilerLabel; });
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

    std::string path = "psa_inline_splitpack_work.csv";
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

} // namespace device::psa_inline_splitpack_work
