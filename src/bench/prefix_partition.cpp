#include "src/bench/prefix_partition.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_partition/PrefixPartition.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/device/prng/PRNG.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <random>
#include <spdlog/spdlog.h>

namespace device::partition_scan {

using weight_type = float;
using Buffers = PrefixPartition<weight_type>::Buffers;
struct NamedConfig {
    std::string name;
    std::string group;
    PrefixPartition<weight_type>::Config config;
    bool writePartition;
    bool flushL2;
};

constexpr bool WRITE_PARTITION = true;

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{//
                .name = "SingleDispatch-RANKED-STRIDED-8",
                .group = "SingleDispatch-RANKED-STRIDED",
                .config = DecoupledPrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SingleDispatch-RANKED-STRIDED-8",
                .group = "SingleDispatch-RANKED-STRIDED",
                .config = DecoupledPrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SingleDispatch-RANKED-STRIDED-4",
                .group = "SingleDispatch-RANKED-STRIDED",
                .config = DecoupledPrefixPartitionConfig(512, 4, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SingleDispatch-RANKED-STRIDED-4",
                .group = "SingleDispatch-RANKED-STRIDED",
                .config = DecoupledPrefixPartitionConfig(512, 4, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SingleDispatch-RANKED-STRIDED-2",
                .group = "SingleDispatch-RANKED-STRIDED",
                .config = DecoupledPrefixPartitionConfig(512, 2, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = false}, //
    NamedConfig{                   //
                .name = "SingleDispatch-RANKED-STRIDED-2",
                .group = "SingleDispatch-RANKED-STRIDED",
                .config = DecoupledPrefixPartitionConfig(512, 2, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = true}, //
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-8",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 8),
                .writePartition = WRITE_PARTITION,
                .flushL2 = false},
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-8",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 8),
                .writePartition = WRITE_PARTITION,
                .flushL2 = true},
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-8",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = false},
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-8",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = true},
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-4",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 4, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = false},
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-4",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 4, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = true},
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-2",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 2, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = false},
    NamedConfig{.name = "BlockWise-RANKED-STRIDED-2",
                .group = "BlockWise-RANKED-STRIDED",
                .config = BlockWisePrefixPartitionConfig(512, 2, BlockScanVariant::RANKED_STRIDED),
                .writePartition = WRITE_PARTITION,
                .flushL2 = true},

};

static constexpr std::size_t N = (1 << 28);
static constexpr std::size_t N_min = (1 << 16);
static constexpr std::size_t ticks = 1000;
static constexpr std::size_t iterations = 100;

struct ConfigResult {
    std::size_t N;
    double latency;         // ms
    double stdVar;          // ms
    double memoryBandwidth; // Gb per second.
    double throughput;      // billion items per second
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
                                       const PrefixPartition<weight_type>::Config& config,
                                       bool flushL2,
                                       bool writePartition) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    PrefixPartition<weight_type> parscan{context, shaderCompiler, config, writePartition};

    Buffers local;
    PhiloxBuffers temp;
    { // Setup
        const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
        assert(resourceExt != nullptr);
        auto alloc = resourceExt->resource_allocator();

        local = Buffers::allocate<weight_type>(alloc, merian::MemoryMappingType::NONE, config, N);
        Buffers stage = Buffers::allocate<weight_type>(
            alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, config, N);

        {

            merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
            cmd->begin();
            Buffers::PivotView<weight_type> stageView{stage.pivot};
            Buffers::PivotView<weight_type> localView{local.pivot};
            stageView.upload<weight_type>(0.5);
            stageView.copyTo(cmd, localView);
            localView.expectComputeRead(cmd);
            cmd->end();
            queue->submit_wait(cmd);
        }

        temp = PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, N);
    }

    ConfigBenchmark results;
    results.entries.reserve(ticks);

    PRNG prng{context, shaderCompiler, PhiloxConfig(512)};
    PRNGBuffers prngBuffers;
    prngBuffers.samples = local.elements;
    PRNGBuffers flushBuffers;
    flushBuffers.samples = temp.samples;

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;

    for (const std::size_t n : host::exp::log10scale<std::size_t>(N_min, N, ticks)) {
        if (n > parscan.maxElementCount()) {
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

                if (flushL2) {
                    prng.run(cmd, flushBuffers, N);

                    cmd->barrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eShaderRead));
                }

                cmd->end();
                queue->submit_wait(cmd);
            }
            {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();
                profiler->start("Build");
                profiler->cmd_start(cmd, "Build");
                parscan.run(cmd, local, n);
                profiler->end();
                profiler->cmd_end(cmd);
                cmd->end();
                queue->submit_wait(cmd);
            }
        }

        profiler->collect(true, true);

        auto report = profiler->get_report();
        auto entry = std::ranges::find_if(report.gpu_report,
                                          [](const auto& entry) { return entry.name == "Build"; });

        double latency = entry->duration;
        double stdVar = entry->std_deviation;

        std::size_t requireTransactionsByte;
        if (writePartition) {
          requireTransactionsByte = (sizeof(weight_type) * 4) * n;
        } else {
          requireTransactionsByte = (sizeof(weight_type) * 3) * n;
        }
        double memoryBandwidth = ((requireTransactionsByte * 1e-9) / (entry->duration * 1e-3));
        double itemsPerSecond = (n / (entry->duration * 1e-3)) / 1e9;

        results.entries.push_back(ConfigResult{.N = n,
                                               .latency = latency,
                                               .stdVar = stdVar,
                                               .memoryBandwidth = memoryBandwidth,
                                               .throughput = itemsPerSecond});
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
            benchmarkConfiguration(context, shaderCompiler, queue, config.config, config.flushL2, config.writePartition);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
        ++i;
    }

    // export

    std::string path = "partition_scan_benchmark.csv";
    host::exp::CSVWriter<9> csv({"N", "method", "group", "latency", "std_derivation", "throughput",
                                 "memory_throughput", "write-partition", "flushL2"},
                                path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, method, r1.configuration.group, r2.latency, r2.stdVar, r2.throughput,
                        r2.memoryBandwidth, r1.configuration.writePartition,
                        r1.configuration.flushL2);
        }
    }
}

} // namespace device::partition_scan
