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
#include "src/device/wrs/alias/psa/pack/scalar/ScalarPack.hpp"
#include "src/device/wrs/alias/psa/pack/subgroup/SubgroupPack.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPack.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPackAllocFlags.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/gen/weight_generator.h"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <fmt/format.h>
#include <random>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unistd.h>

namespace device::psa_inline_splitpack_splitSize {

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
                .name = "SplitPack-1",
                .group = "SplitPack-1",
                .packsPerSubgroup = 32,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-2",
                .group = "SplitPack-2",
                .packsPerSubgroup = 16,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-4",
                .group = "SplitPack-4",
                .packsPerSubgroup = 8,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-8",
                .group = "SplitPack-8",
                .packsPerSubgroup = 4,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-16",
                .group = "SplitPack-16",
                .packsPerSubgroup = 2,
                .workgroupSize = 128,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SplitPack-32",
                .group = "SplitPack-32",
                .packsPerSubgroup = 1,
                .workgroupSize = 128,
                .flushL2 = true}, //
};

/* static constexpr std::size_t N = 1e6; */
static constexpr std::size_t w = 10;
static constexpr host::Distribution weight_distribution = host::Distribution::SEEDED_RANDOM_UNIFORM;
static constexpr std::size_t maxOccupantInvoc = 70656;
static constexpr std::size_t min_SplitSize = 2;
static constexpr std::size_t max_SplitSize = 128;
static constexpr std::size_t step = 1;

static constexpr std::size_t iterations = 100;
static constexpr std::size_t flushSize = 1e7;

static constexpr MeanConfig meanConfig = AtomicMeanConfig();
static constexpr PrefixPartitionConfig prefixPartitionConfig = DecoupledPrefixPartitionConfig();

static constexpr bool writeElements = false;

struct ConfigResult {
    std::size_t N;
    std::size_t splitSize;
    std::size_t threadsPerPack;
    std::size_t workgroupSize;
    double latency; // ms
    double stdVar;  // ms
    bool writeElements;
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
    std::size_t maxK = maxInvoc / threadsPerPack;
    std::size_t maxN = maxK * max_SplitSize;

    PhiloxBuffers flush =
        PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, flushSize);

    PSABuffers psaBuffers = PSABuffers::allocate(
        alloc, merian::MemoryMappingType::NONE,
        PSAConfig(meanConfig, prefixPartitionConfig,
                  InlineSplitPackConfig(min_SplitSize, packsPerSubgroup), writeElements),
        maxN);
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
                                                 writeElements};

    ConfigBenchmark results;

    PRNG prng{context, shaderCompiler, PhiloxConfig(weight_distribution)};
    PRNGBuffers prngBuffers;
    prngBuffers.samples = weights.samples;
    PRNGBuffers flushBuffers;
    flushBuffers.samples = flush.samples;

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 128 * iterations);
    query_pool->reset();
    profiler->set_query_pool(query_pool);
    for (std::size_t splitSize = min_SplitSize; splitSize <= max_SplitSize; splitSize += step) {
        /* fmt::println("Progess : {}  ({})", */
        /*              (splitSize - min_SplitSize) / (float)(max_SplitSize - min_SplitSize), */
        /*              splitSize); */

        std::size_t invoc = maxOccupantInvoc * w;
        std::size_t K = invoc / threadsPerPack;
        std::size_t n = K * splitSize;
        /* fmt::println("n = {}, maxN = {}, K = {}, maxK = {}", n, maxN, K, maxK); */
        if (n > (1 << 28)) {
            break;
        }
        if (n > maxN) {
            throw std::runtime_error("Well that's wrong");
        }

        SplitPack splitPack{context, shaderCompiler,
                            InlineSplitPackConfig(splitSize, packsPerSubgroup, workgroupSize),
                            writeElements};

        std::size_t threadsPerSubproblem =
            context->physical_device.physical_device_subgroup_properties.subgroupSize /
            packsPerSubgroup;
        if (threadsPerSubproblem >= splitSize) {
            /* fmt::println("Skipping"); */
            /* std::cout << "\x1b[2K"; // Delete current line */
            /* std::cout << "\x1b[1A"  // Move cursor up one */
            /*           << "\x1b[2K"; // Delete the entire line */
            /* std::cout << "\r";      // Resume the cursor at beginning of line */
            continue; // doesn't make any sense
        }

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

        std::string label = fmt::format("SplitPack-{}", splitSize);
        for (std::size_t i = 0; i < iterations; ++i) {
            if (flushL2) {
                /* merian::CommandBufferHandle cmd =
                 * std::make_shared<merian::CommandBuffer>(cmdPool); */
                /* cmd->begin(); */
                /*  */
                /* prng.run(cmd, flushBuffers, flushSize, dist(rng)); */
                /* cmd->barrier(vk::PipelineStageFlagBits::eComputeShader, */
                /*              vk::PipelineStageFlagBits::eComputeShader, */
                /*              prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                 */
                /*                                                  vk::AccessFlagBits::eShaderRead));
                 */
                /*  */
                /* cmd->end(); */
                /* queue->submit_wait(cmd); */
            }
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
                {
                    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, label);
                    splitPack.run(cmd, splitPackBuffers, n);
                }

                cmd->end();
                queue->submit_wait(cmd);
            }
        }
        profiler->collect(true, true);

        auto report = profiler->get_report();
        auto entry = std::ranges::find_if(report.gpu_report,
                                          [&](const auto& entry) { return entry.name == label; });

        assert(entry != report.gpu_report.end());
        double latency = entry->duration;
        double stdVar = entry->std_deviation;

        SPDLOG_INFO(fmt::format("\nsplitSize={}\n"
                                "n = {}\n"
                                "SplitPack-{}\n"
                                "throughput={}G/s\n"
                                "gpu-time={}ms\n"
                                "latency={}\n",
                                splitSize,                   //
                                n,                           //
                                threadsPerPack,              //
                                n / (latency * 1e-3) * 1e-9, //
                                report.gpu_total(),          //
                                latency                      //

                                ));

        results.entries.push_back(ConfigResult{
            .N = n,
            .splitSize = splitSize,
            .threadsPerPack = threadsPerPack,
            .workgroupSize = workgroupSize,
            .latency = latency,
            .stdVar = stdVar,
            .writeElements = false,
        });

        /* std::cout << "\x1b[2K"; // Delete current line */
        /* std::cout << "\x1b[1A"  // Move cursor up one */
        /*           << "\x1b[2K"; // Delete the entire line */
        /* std::cout << "\r";      // Resume the cursor at beginning of line */
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
    SPDLOG_INFO("Benchmarking PSA-InlineSplitPack (average subgroup latency based on split size)");
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

    std::string path = "psa_inline_splitpack_benchmark_splitSizes.csv";
    host::exp::CSVWriter<10> csv({"N", "splitSize", "method", "group", "latency", "std_derivation",
                                  "flushL2", "workgroupSize", "threadsPerPack", "writeElements"},
                                 path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.splitSize, method, r1.configuration.group, r2.latency, r2.stdVar,
                        r1.configuration.flushL2, r2.workgroupSize, r2.threadsPerPack,
                        r2.writeElements);
        }
    }
}

} // namespace device::psa_splitpack_inline
