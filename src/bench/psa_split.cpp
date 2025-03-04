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

namespace device::psa_split {

using weight_type = float;
struct NamedConfig {
    std::string name;
    std::string group;
    bool flushL2;
};

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{//
                .name = "ScalarSplit-512",
                .group = "ScalarSplit",
                .flushL2 = true}, //
};

static constexpr std::size_t min_SplitSize = 2;
static constexpr std::size_t max_SplitSize = 128;
static constexpr std::size_t step = 1;

static constexpr std::size_t iterations = 1000;
static constexpr std::size_t flushSize = 1e7;

static constexpr std::size_t maxOccupantSubgroupsPerSM = 64; // ada lovelace architecture
static constexpr std::size_t adaLovelaceSubgroupSize = 32;
static constexpr std::size_t AmountOfSMs = 46; // RTX 4070
static constexpr std::size_t maxOccupantThreads =
    maxOccupantSubgroupsPerSM * AmountOfSMs * adaLovelaceSubgroupSize; // = 94208

static constexpr float gpuOccupancyFactor = 10;

static constexpr MeanConfig meanConfig = AtomicMeanConfig();
static constexpr PrefixPartitionConfig prefixPartitionConfig = DecoupledPrefixPartitionConfig();

struct ConfigResult {
    std::size_t N;
    std::size_t splitSize;
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
                                       bool flushL2) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
    assert(resourceExt != nullptr);
    auto alloc = resourceExt->resource_allocator();

    std::size_t maxK = maxOccupantThreads * gpuOccupancyFactor;
    std::size_t maxN = maxK * max_SplitSize;

    PhiloxBuffers weights = PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, maxN);
    PhiloxBuffers flush =
        PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, flushSize);

    MeanBuffers meanBuffers =
        MeanBuffers::allocate<weight_type>(alloc, merian::MemoryMappingType::NONE, meanConfig, maxN,
                                           MeanAllocFlags::ALLOC_ONLY_OUTPUT);
    meanBuffers.elements = weights.samples;
    PrefixPartitionBuffers prefixPartitionBuffers = PrefixPartitionBuffers::allocate<weight_type>(
        alloc, merian::MemoryMappingType::NONE, prefixPartitionConfig, maxN,
        PrefixPartitionAllocFlags::ALLOC_ONLY_OUTPUT &
            ~PrefixPartitionAllocFlags::ALLOC_PARTITION_ELEMENTS);
    prefixPartitionBuffers.elements = weights.samples;
    prefixPartitionBuffers.pivot = meanBuffers.mean;

    SplitBuffers splitBuffers = SplitBuffers::allocate(alloc, merian::MemoryMappingType::NONE, maxN,
                                                       maxK, SplitAllocFlags::ALLOC_ONLY_OUTPUTS);
    splitBuffers.partitionPrefix = prefixPartitionBuffers.partitionPrefix;
    splitBuffers.mean = meanBuffers.mean;
    splitBuffers.heavyCount = prefixPartitionBuffers.heavyCount;

    Mean<weight_type> mean{context, shaderCompiler, meanConfig};
    PrefixPartition<weight_type> prefixPartition{context, shaderCompiler, prefixPartitionConfig,
                                                 false};

    ConfigBenchmark results;

    PRNG prng{context, shaderCompiler, PhiloxConfig(512)};
    PRNGBuffers prngBuffers;
    prngBuffers.samples = weights.samples;
    PRNGBuffers flushBuffers;
    flushBuffers.samples = flush.samples;

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;

    for (std::size_t splitSize = min_SplitSize; splitSize <= max_SplitSize; splitSize += step) {
        Split split{context, shaderCompiler, ScalarSplitConfig(splitSize)};

        merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
        merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
            std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 4 * iterations);
        query_pool->reset();
        profiler->set_query_pool(query_pool);

        host::glsl::uint N = splitSize * maxK;

        for (std::size_t i = 0; i < iterations; ++i) {
            { // Generate uni
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();
                prng.run(cmd, prngBuffers, N, dist(rng));
                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                mean.run(cmd, meanBuffers, N);

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             meanBuffers.mean->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead));

                prefixPartition.run(cmd, prefixPartitionBuffers, N);

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
                    prng.run(cmd, flushBuffers, flushSize, dist(rng));
                    cmd->barrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eShaderRead));
                }

                cmd->end();
                queue->submit_wait(cmd); // wait idle
            }
            {

                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();

                profiler->start("Split");
                profiler->cmd_start(cmd, "Split");
                split.run(cmd, splitBuffers, N);
                profiler->end();
                profiler->cmd_end(cmd);

                cmd->end();
                queue->submit_wait(cmd);
            }
            profiler->collect(true, true);
        }

        auto report = profiler->get_report();
        auto entry = std::ranges::find_if(report.gpu_report,
                                          [](const auto& entry) { return entry.name == "Split"; });
        double latency = entry->duration;
        double stdVar = entry->std_deviation;

        results.entries.push_back(ConfigResult{
            .N = N,
            .splitSize = splitSize,
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
    SPDLOG_INFO("Benchmarking PSA-Split (average subgroup latency based on split size)");
    for (const auto& config : CONFIGURATIONS) {
        SPDLOG_INFO(
            "[{}%] Benchmarking {}",
            (i / static_cast<float>(sizeof(CONFIGURATIONS) / (float)sizeof(CONFIGURATIONS[0]))) *
                100.0f,
            config.name);
        auto configBenchmark =
            benchmarkConfiguration(context, shaderCompiler, queue, config.flushL2);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
        ++i;
    }

    // export

    std::string path = "psa_split_benchmark_fixed_N.csv";
    host::exp::CSVWriter<7> csv(
        {"N", "splitSize", "method", "group", "latency", "std_derivation", "flushL2"}, path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.splitSize, method, r1.configuration.group, r2.latency, r2.stdVar,
                        r1.configuration.flushL2);
        }
    }
}

} // namespace device::psa_split
