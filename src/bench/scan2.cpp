
#include "./wrs.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/PrefixSum.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include "src/host/gen/weight_generator.h"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <spdlog/spdlog.h>

namespace device::scan {

using weight_type = float;
using Buffers = PrefixSum<weight_type>::Buffers;
struct NamedConfig {
    std::string name;
    BlockScan<weight_type>::Config config;
};

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{
        .name = "Ranked",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Ranked",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "RegisterScan",
        .config = BlockScanConfig(512, 2, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "RegisterScan",
        .config = BlockScanConfig(512, 4, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "RegisterScan",
        .config = BlockScanConfig(512, 8, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "KoggeStoneShfl",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "KoggeStoneShfl",
        .config = BlockScanConfig(512, 2, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "KoggeStoneShfl",
        .config = BlockScanConfig(512, 4, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "KoggeStoneShfl",
        .config = BlockScanConfig(512, 8, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Raking",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Raking",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Raking",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Ranked-Strided",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Ranked-Strided",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Ranked-Strided",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
    NamedConfig{
        .name = "Sequential-Scan",
        .config = BlockScanConfig(512, 1, BlockScanVariant::RANKED, 1, false),
    }, //*/
};

static constexpr std::size_t N = (1 << 21);

struct ConfigBenchmark {
    std::size_t blockSize; // 512, 1024, 2048, 4096
    double latency;
    double throughput;
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
                                       const PrefixSum<weight_type>::Config& config) {
    ConfigBenchmark results;

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
        auto configBenchmark =
            benchmarkConfiguration(context, shaderCompiler, queue, config.config);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
    }

    // export

    std::string path = "scan_benchmark.csv";
    host::exp::CSVWriter<6> csv({"method", "latency", "throughput"}, path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(method, r2.latency, r2.throughput);
        }
    }
}

} // namespace device::scan
