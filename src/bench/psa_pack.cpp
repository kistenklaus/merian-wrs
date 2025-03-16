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
#include "src/device/wrs/alias/psa/pack/subgroup/SubgroupPack.hpp"
#include "src/device/wrs/alias/psa/split/Split.hpp"
#include "src/device/wrs/alias/psa/split/SplitAllocFlags.hpp"
#include "src/device/wrs/alias/psa/split/scalar/ScalarSplit.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/gen/weight_generator.h"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <random>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace device::psa_pack {

using weight_type = float;
struct NamedConfig {
    std::string name;
    std::string group;
    std::size_t packsPerSubgroup;
    std::size_t workgroupSize;
    bool flushL2;
};

static const NamedConfig CONFIGURATIONS[] = {
    /* NamedConfig{// */
    /*            .name = "ScalarPack-512", */
    /*            .group = "Pack-1", */
    /*            .packsPerSubgroup = 32, */
    /*            .workgroupSize = 128, */
    /*            .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*            .name = "ScalarPack-512", */
    /*            .group = "Pack-2", */
    /*            .packsPerSubgroup = 16, */
    /*            .workgroupSize = 128, */
    /*            .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*            .name = "ScalarPack-512", */
    /*            .group = "Pack-4", */
    /*            .packsPerSubgroup = 8, */
    /*            .workgroupSize = 128, */
    /*            .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*            .name = "ScalarPack-512", */
    /*            .group = "Pack-8", */
    /*            .packsPerSubgroup = 4, */
    /*            .workgroupSize = 128, */
    /*            .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*            .name = "ScalarPack-512", */
    /*            .group = "Pack-16", */
    /*            .packsPerSubgroup = 2, */
    /*            .workgroupSize = 128, */
    /*            .flushL2 = false}, // */
    /* NamedConfig{                   // */
    /*            .name = "ScalarPack-512", */
    /*            .group = "Pack-32", */
    /*            .packsPerSubgroup = 1, */
    /*            .workgroupSize = 128, */
    /*            .flushL2 = false}, // */
    //
    //
    //
    NamedConfig{//
                .name = "ScalarPack-512",
                .group = "Pack-1",
                .packsPerSubgroup = 32,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "ScalarPack-512",
                .group = "Pack-2",
                .packsPerSubgroup = 16,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "ScalarPack-512",
                .group = "Pack-4",
                .packsPerSubgroup = 8,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "ScalarPack-512",
                .group = "Pack-8",
                .packsPerSubgroup = 4,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "ScalarPack-512",
                .group = "Pack-16",
                .packsPerSubgroup = 2,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "ScalarPack-512",
                .group = "Pack-32",
                .packsPerSubgroup = 1,
                .workgroupSize = 128,
                .flushL2 = true}, //
};

/* static constexpr std::size_t N = 1e6; */
static constexpr std::size_t w = 4;
static constexpr host::Distribution weight_distribution =
    host::Distribution::SEEDED_RANDOM_EXPONENTIAL;
static constexpr std::size_t maxOccupantInvoc = 70656;
static constexpr std::size_t min_SplitSize = 2;
static constexpr std::size_t max_SplitSize = 128;
static constexpr std::size_t step = 1;

static constexpr std::size_t iterations = 250;
static constexpr std::size_t flushSize = 1e7;

static constexpr MeanConfig meanConfig = AtomicMeanConfig();
static constexpr PrefixPartitionConfig prefixPartitionConfig = DecoupledPrefixPartitionConfig();

struct ConfigResult {
    std::size_t N;
    std::size_t splitSize;
    std::size_t threadsPerPack;
    std::size_t workgroupSize;
    double latency;      // ms
    double stdVar;       // ms
    double splitLatency; // ms
    double splitStd;     // ms
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
                                       std::size_t packsPerSubgroup,
                                       std::size_t workgroupSize,
                                       bool flushL2) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
    assert(resourceExt != nullptr);
    auto alloc = resourceExt->resource_allocator();

    std::size_t threadsPerPack =
        context->physical_device.physical_device_subgroup_properties.subgroupSize /
        packsPerSubgroup;
    std::size_t maxInvoc = maxOccupantInvoc * w;
    std::size_t maxK = (maxInvoc + threadsPerPack - 1) / threadsPerPack;
    std::size_t maxN = std::min<std::size_t>(maxK * max_SplitSize, 1 << 28);

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

    ConfigBenchmark results;

    PRNG prng{context, shaderCompiler, PhiloxConfig(weight_distribution)};
    PRNGBuffers prngBuffers;
    prngBuffers.samples = weights.samples;
    PRNGBuffers flushBuffers;
    flushBuffers.samples = flush.samples;

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;

    for (std::size_t splitSize = min_SplitSize; splitSize <= max_SplitSize; splitSize += step) {
        fmt::println("Progess : {}  ({})",
                     (splitSize - min_SplitSize) / (float)(max_SplitSize - min_SplitSize),
                     splitSize);

        std::size_t invoc = maxOccupantInvoc * w;
        std::size_t K = (invoc + threadsPerPack - 1) / threadsPerPack;
        std::size_t n = K * splitSize;
        /* fmt::println("n = {}, maxN = {}, K = {}, maxK = {}", n, maxN, K, maxK); */
        if (n > (1 << 28)) {
            break;
        }
        if (n > maxN) {
            throw std::runtime_error("Well that's wrong");
        }

        Split split{context, shaderCompiler, ScalarSplitConfig(splitSize, 128)};

        PackConfig packConfig = [&]() -> PackConfig {
            if (packsPerSubgroup ==
                context->physical_device.physical_device_subgroup_properties.subgroupSize) {
                return ScalarPackConfig(splitSize, workgroupSize);
            } else {
                return SubgroupPackConfig(splitSize, packsPerSubgroup, workgroupSize);
            }
        }();
        Pack pack{context, shaderCompiler, packConfig, false};

        std::size_t threadsPerSubproblem =
            context->physical_device.physical_device_subgroup_properties.subgroupSize /
            packsPerSubgroup;
        if (threadsPerSubproblem >= splitSize) {
            /* fmt::println("Skipping"); */
            std::cout << "\x1b[2K"; // Delete current line
            std::cout << "\x1b[1A"  // Move cursor up one
                      << "\x1b[2K"; // Delete the entire line
            std::cout << "\r";      // Resume the cursor at beginning of line
            continue;               // doesn't make any sense
        }

        merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
        merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
            std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 4 * iterations);
        query_pool->reset();
        profiler->set_query_pool(query_pool);
        { // Generate uni
            merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
            cmd->begin();
            prng.run(cmd, prngBuffers, n, dist(rng));
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

            prng.run(cmd, prngBuffers, n, dist(rng));
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eComputeShader,
                         prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                             vk::AccessFlagBits::eShaderRead));
            if (flushL2) {
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();

                prng.run(cmd, flushBuffers, flushSize, dist(rng));
                cmd->barrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {
                        prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eShaderRead),
                        flushBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                             vk::AccessFlagBits::eShaderRead),

                    });

                cmd->end();
                queue->submit_wait(cmd);
            }
            {

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

                profiler->start("Split");
                profiler->cmd_start(cmd, "Split");

                split.run(cmd, splitBuffers, n);

                profiler->end();
                profiler->cmd_end(cmd);

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             splitBuffers.splits->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                profiler->start("Pack");
                profiler->cmd_start(cmd, "Pack");

                pack.run(cmd, packBuffers, n);

                profiler->end();
                profiler->cmd_end(cmd);
            }
            cmd->end();
            queue->submit_wait(cmd);
            profiler->collect(true, true);
        }

        /* profiler->collect(true, true); */
        /* SPDLOG_INFO(fmt::format("Profiler results: \n{}", merian::Profiler::get_report_str(
         */
        /*                                                       profiler->get_report()))); */

        auto report = profiler->get_report();
        auto entry = std::ranges::find_if(report.gpu_report,
                                          [](const auto& entry) { return entry.name == "Pack"; });

        auto entrySplit = std::ranges::find_if(
            report.gpu_report, [](const auto& entry) { return entry.name == "Split"; });
        assert(entry != report.gpu_report.end());
        double latency = entry->duration;
        double stdVar = entry->std_deviation;
        double splitLatency = entrySplit->duration;
        double splitStd = entrySplit->std_deviation;

        results.entries.push_back(ConfigResult{
            .N = n,
            .splitSize = splitSize,
            .threadsPerPack = threadsPerPack,
            .workgroupSize = workgroupSize,
            .latency = latency,
            .stdVar = stdVar,
            .splitLatency = splitLatency,
            .splitStd = splitStd,
        });

        std::cout << "\x1b[2K"; // Delete current line
        std::cout << "\x1b[1A"  // Move cursor up one
                  << "\x1b[2K"; // Delete the entire line
        std::cout << "\r";      // Resume the cursor at beginning of line
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
    SPDLOG_INFO("Benchmarking PSA-Pack (average subgroup latency based on split size)");
    for (const auto& config : CONFIGURATIONS) {
        SPDLOG_INFO(
            "[{}%] Benchmarking {} ({})",
            (i / static_cast<float>(sizeof(CONFIGURATIONS) / (float)sizeof(CONFIGURATIONS[0]))) *
                100.0f,
            config.group, config.name);
        auto configBenchmark =
            benchmarkConfiguration(context, shaderCompiler, queue, config.packsPerSubgroup,
                                   config.workgroupSize, config.flushL2);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
        ++i;
    }

    // export

    std::string path = "psa_pack_benchmark_splitSizes.csv";
    host::exp::CSVWriter<11> csv({"N", "splitSize", "method", "group", "latency", "std_derivation",
                                  "splitLatency", "splitStd", "flushL2", "workgroupSize",
                                  "threadsPerPack"},
                                 path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.splitSize, method, r1.configuration.group, r2.latency, r2.stdVar,
                        r2.splitLatency, r2.splitStd, r1.configuration.flushL2, r2.workgroupSize,
                        r2.threadsPerPack);
        }
    }
}

} // namespace device::psa_pack
