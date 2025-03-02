#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/device/prng/PRNG.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include <algorithm>
#include <csignal>
#include <cwchar>
#include <fmt/base.h>
#include <random>
#include <spdlog/spdlog.h>

namespace device::block_scan {

using weight_type = float;
using Buffers = BlockScan<weight_type>::Buffers;
struct NamedConfig {
    std::string name;
    std::string methodGroup;
    BlockScanConfig config;
};

static constexpr bool flushL2 = false;

static const NamedConfig CONFIGURATIONS[] = {
    //NamedConfig{.name = "glSubgroupInclusiveAdd",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 1, false)},
    ////*/
    //NamedConfig{.name = "KoggeStoneShfl",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      8,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      1,
    //                                      false)}, //*/
    //NamedConfig{.name = "glSubgroupInclusiveAdd",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 1, false)},
    ////*
    //NamedConfig{.name = "KoggeStoneShfl",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      8,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      1,
    //                                      false)}, //*/
    //NamedConfig{.name = "glSubgroupInclusiveAdd-2",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 2, BlockScanVariant::RANKED_STRIDED, 1, false)},
    ////*/
    //NamedConfig{.name = "KoggeStoneShfl-2",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      2,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      1,
    //                                      false)}, //*/
    //NamedConfig{.name = "glSubgroupInclusiveAdd-1",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED_STRIDED, 1, false)},
    ////*
    //NamedConfig{.name = "KoggeStoneShfl-1",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      1,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      1,
    //                                      false)}, //*/
    NamedConfig{.name = "RAKING-2-1",
                .methodGroup = "RAKING",
                .config = BlockScanConfig(512, 2, BlockScanVariant::RAKING, 1, false)}, //*/
    NamedConfig{.name = "RAKING-2-1",
                .methodGroup = "RAKING",
                .config = BlockScanConfig(512, 2, BlockScanVariant::RAKING, 1, false)}, //*/
    NamedConfig{.name = "RAKING-4-1",
                .methodGroup = "RAKING",
                .config = BlockScanConfig(512, 4, BlockScanVariant::RAKING, 1, false)}, //*/
    NamedConfig{.name = "RAKING-4-2",
                .methodGroup = "RAKING",
                .config = BlockScanConfig(512, 4, BlockScanVariant::RAKING, 2, false)}, //*/
    NamedConfig{.name = "RAKING-2-2",
                .methodGroup = "RAKING",
                .config = BlockScanConfig(512, 2, BlockScanVariant::RAKING, 2, false)}, //*/
    NamedConfig{.name = "RAKING-1-2",
                .methodGroup = "RAKING",
                .config = BlockScanConfig(512, 1, BlockScanVariant::RAKING, 2, false)}, //*/
    NamedConfig{.name = "RAKING-2-1",
                .methodGroup = "RAKING",
                .config = BlockScanConfig(512, 2, BlockScanVariant::RAKING, 1, false)}, //*/
    //NamedConfig{.name = "RANKED-2-1",
    //            .methodGroup = "RANKED",
    //            .config = BlockScanConfig(512, 2, BlockScanVariant::RANKED, 1, false)}, //*/
    //NamedConfig{.name = "RANKED-4-1",
    //            .methodGroup = "RANKED",
    //            .config = BlockScanConfig(512, 4, BlockScanVariant::RANKED, 1, false)}, //*/
    //NamedConfig{.name = "RANKED-4-2",
    //            .methodGroup = "RANKED",
    //            .config = BlockScanConfig(512, 4, BlockScanVariant::RANKED, 2, false)}, //*/
    //NamedConfig{.name = "RANKED-4-4",
    //            .methodGroup = "RANKED",
    //            .config = BlockScanConfig(512, 4, BlockScanVariant::RANKED, 4, false)}, //*/
    //NamedConfig{.name = "RANKED-2-2",
    //            .methodGroup = "RANKED",
    //            .config = BlockScanConfig(512, 2, BlockScanVariant::RANKED, 2, false)}, //*/
    //NamedConfig{.name = "RANKED-1-2",
    //            .methodGroup = "RANKED",
    //            .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 2, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-2-1",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 2, BlockScanVariant::RANKED_STRIDED, 1, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-4-1",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 4, BlockScanVariant::RANKED_STRIDED, 1, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-4-2",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 4, BlockScanVariant::RANKED_STRIDED, 2, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-2-2",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 2, BlockScanVariant::RANKED_STRIDED, 2, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-1-2",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED_STRIDED, 2, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-8-1",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 1, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-8-2",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 2, false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-8-4",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 4, false)}, //*/

    NamedConfig{.name = "RAKING-2-1-KoggeStone",
                .methodGroup = "RAKING",
                .config =
                    BlockScanConfig(512,
                                    2,
                                    BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                                    1,
                                    false)}, //*/
    NamedConfig{.name = "RAKING-2-1-KoggeStone",
                .methodGroup = "RAKING",
                .config =
                    BlockScanConfig(512,
                                    2,
                                    BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                                    1,
                                    false)}, //*/
    NamedConfig{.name = "RAKING-4-1-KoggeStone",
                .methodGroup = "RAKING",
                .config =
                    BlockScanConfig(512,
                                    4,
                                    BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                                    1,
                                    false)}, //*/
    NamedConfig{.name = "RAKING-4-2-KoggeStone",
                .methodGroup = "RAKING",
                .config =
                    BlockScanConfig(512,
                                    4,
                                    BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                                    2,
                                    false)}, //*/
    NamedConfig{.name = "RAKING-2-2-KoggeStone",
                .methodGroup = "RAKING",
                .config =
                    BlockScanConfig(512,
                                    2,
                                    BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                                    2,
                                    false)}, //*/
    NamedConfig{.name = "RAKING-1-2-KoggeStone",
                .methodGroup = "RAKING",
                .config =
                    BlockScanConfig(512,
                                    1,
                                    BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                                    2,
                                    false)}, //*/
    NamedConfig{.name = "RAKING-2-1-KoggeStone",
                .methodGroup = "RAKING",
                .config =
                    BlockScanConfig(512,
                                    2,
                                    BlockScanVariant::RAKING | BlockScanVariant::SUBGROUP_SCAN_SHFL,
                                    1,
                                    false)}, //*/
    //NamedConfig{.name = "RANKED-2-1-KoggeStone",
    //            .methodGroup = "RANKED",
    //            .config =
    //                BlockScanConfig(512,
    //                                2,
    //                                BlockScanVariant::RANKED | BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                1,
    //                                false)}, //*/
    //NamedConfig{.name = "RANKED-4-1-KoggeStone",
    //            .methodGroup = "RANKED",
    //            .config =
    //                BlockScanConfig(512,
    //                                4,
    //                                BlockScanVariant::RANKED | BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                1,
    //                                false)}, //*/
    //NamedConfig{.name = "RANKED-4-2-KoggeStone",
    //            .methodGroup = "RANKED",
    //            .config =
    //                BlockScanConfig(512,
    //                                4,
    //                                BlockScanVariant::RANKED | BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                2,
    //                                false)}, //*/
    //NamedConfig{.name = "RANKED-4-4-KoggeStone",
    //            .methodGroup = "RANKED",
    //            .config =
    //                BlockScanConfig(512,
    //                                4,
    //                                BlockScanVariant::RANKED | BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                4,
    //                                false)}, //*/
    //NamedConfig{.name = "RANKED-2-2-KoggeStone",
    //            .methodGroup = "RANKED",
    //            .config =
    //                BlockScanConfig(512,
    //                                2,
    //                                BlockScanVariant::RANKED | BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                2,
    //                                false)}, //*/
    //NamedConfig{.name = "RANKED-1-2-KoggeStone",
    //            .methodGroup = "RANKED",
    //            .config = BlockScanConfig(512,
    //                                      1,
    //                                      BlockScanVariant::RANKED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      2,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-2-1-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      2,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      1,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-4-1-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      4,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      1,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-4-2-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      4,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      2,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-2-2-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      2,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      2,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-1-2-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      1,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      2,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-8-1-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      8,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      1,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-8-2-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      8,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      2,
    //                                      false)}, //*/
    //NamedConfig{.name = "RANKED-STRIDED-8-4-KoggeStone",
    //            .methodGroup = "RANKED-STRIDED",
    //            .config = BlockScanConfig(512,
    //                                      8,
    //                                      BlockScanVariant::RANKED_STRIDED |
    //                                          BlockScanVariant::SUBGROUP_SCAN_SHFL,
    //                                      4,
    //                                      false)}, //*/
};

static constexpr std::size_t N = (1 << 29);
static constexpr std::size_t N_min = (1 << 16);
static constexpr std::size_t ticks = 1000;
static constexpr std::size_t iterations = 50;

struct ConfigResult {
    std::size_t N;
    std::size_t blockSize;
    bool seq;
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
                                       const BlockScanConfig& config) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    BlockScan<weight_type> scan{context, shaderCompiler, config};

    Buffers local;

    PhiloxBuffers temp;
    { // Setup
        const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
        assert(resourceExt != nullptr);
        auto alloc = resourceExt->resource_allocator();

        local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, N, 1);
        temp = PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, N);
    }

    ConfigBenchmark results;
    results.entries.reserve(ticks);

    PRNG prng(context, shaderCompiler, PhiloxConfig(512));
    PRNGBuffers prngBuffers;
    prngBuffers.samples = local.elements;
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

        for (std::size_t i = 0; i < iterations; ++i) {
            {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();

                prng.run(cmd, prngBuffers, n, dist(rng));

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             local.elements->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eShaderWrite));

                if (flushL2) {
                    prng.run(cmd, flushBuffers, N);
                    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 local.elements->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                vk::AccessFlagBits::eShaderWrite));
                }

                cmd->end();
                queue->submit_wait(cmd);
            }

            {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);

                cmd->begin();
                profiler->start("Build");
                profiler->cmd_start(cmd, "Build");
                scan.run(cmd, local, n);
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

        std::size_t requireTransactionsByte = 2 * n * sizeof(weight_type);
        double memoryBandwidth = ((requireTransactionsByte * 1e-9) / (entry->duration * 1e-3));
        double itemsPerSecond = (n / (entry->duration * 1e-3)) / 1e9;

        results.entries.push_back(
            ConfigResult{.N = n,
                         .blockSize = static_cast<std::size_t>(config.blockSize()),
                         .seq = config.sequentialScanLength > 1,
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

    std::size_t i = 0;
    BenchmarkResults results;
    for (const auto& config : CONFIGURATIONS) {
        SPDLOG_INFO(
            "[{}%] Benchmarking {}",
            (i / static_cast<float>(sizeof(CONFIGURATIONS) / (float)sizeof(CONFIGURATIONS[0]))) *
                100.0f,
            config.name);
        auto configBenchmark =
            benchmarkConfiguration(context, shaderCompiler, queue, config.config);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
        ++i;
    }

    // export

    std::string path = "block_scan_benchmark.csv";
    host::exp::CSVWriter<9> csv({"N", "block_size", "seq", "method", "group", "latency",
                                 "std_derivation", "throughput", "memory_throughput"},
                                path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.blockSize, r2.seq, method, r1.configuration.methodGroup,
                        r2.latency, r2.stdVar, r2.throughput, r2.memoryBandwidth);
        }
    }
}

} // namespace device::block_scan
