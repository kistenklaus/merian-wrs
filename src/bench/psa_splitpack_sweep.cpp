#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "src/host/export/logscale.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/mean/Mean.hpp"
#include "src/device/mean/atomic/AtomicMean.hpp"
#include "src/device/prefix_partition/PrefixPartition.hpp"
#include "src/device/prng/PRNG.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include "src/device/wrs/alias/psa/PSA.hpp"
#include "src/device/wrs/alias/psa/splitpack/SplitPack.hpp"
#include "src/device/wrs/alias/psa/splitpack/inline/InlineSplitPack.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/gen/weight_generator.h"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <fmt/format.h>
#include <random>
#include <spdlog/spdlog.h>
#include <unistd.h>

namespace device::psa_splitpack_sweep {

using weight_type = float;
struct NamedConfig {
    std::string name;
    std::string group;
    std::size_t packsPerSubgroup;
    std::size_t workgroupSize;
    bool serial;
    bool flushL2;
};

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{//
                .name = "InlineSplitPack-1",
                .group = "InlineSplitPack-1",
                .packsPerSubgroup = 32,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-2",
                .group = "InlineSplitPack-2",
                .packsPerSubgroup = 16,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-4",
                .group = "InlineSplitPack-4",
                .packsPerSubgroup = 8,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-8",
                .group = "InlineSplitPack-8",
                .packsPerSubgroup = 4,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-16",
                .group = "InlineSplitPack-16",
                .packsPerSubgroup = 2,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-32",
                .group = "InlineSplitPack-32",
                .packsPerSubgroup = 1,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, 
    NamedConfig{//
                .name = "SerialSplitPack-1",
                .group = "SerialSplitPack-1",
                .packsPerSubgroup = 32,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-2",
                .group = "SerialSplitPack-2",
                .packsPerSubgroup = 16,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-4",
                .group = "SerialSplitPack-4",
                .packsPerSubgroup = 8,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-8",
                .group = "SerialSplitPack-8",
                .packsPerSubgroup = 4,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-16",
                .group = "SerialSplitPack-16",
                .packsPerSubgroup = 2,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = true}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-32",
                .group = "SerialSplitPack-32",
                .packsPerSubgroup = 1,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = true}, 

    NamedConfig{//
                .name = "InlineSplitPack-1",
                .group = "InlineSplitPack-1",
                .packsPerSubgroup = 32,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-2",
                .group = "InlineSplitPack-2",
                .packsPerSubgroup = 16,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-4",
                .group = "InlineSplitPack-4",
                .packsPerSubgroup = 8,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-8",
                .group = "InlineSplitPack-8",
                .packsPerSubgroup = 4,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-16",
                .group = "InlineSplitPack-16",
                .packsPerSubgroup = 2,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "InlineSplitPack-32",
                .group = "InlineSplitPack-32",
                .packsPerSubgroup = 1,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, 
    NamedConfig{//
                .name = "SerialSplitPack-1",
                .group = "SerialSplitPack-1",
                .packsPerSubgroup = 32,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-2",
                .group = "SerialSplitPack-2",
                .packsPerSubgroup = 16,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-4",
                .group = "SerialSplitPack-4",
                .packsPerSubgroup = 8,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-8",
                .group = "SerialSplitPack-8",
                .packsPerSubgroup = 4,
                .workgroupSize = 128,
                .serial = false,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-16",
                .group = "SerialSplitPack-16",
                .packsPerSubgroup = 2,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = false}, //
    NamedConfig{                  //
                .name = "SerialSplitPack-32",
                .group = "SerialSplitPack-32",
                .packsPerSubgroup = 1,
                .workgroupSize = 128,
                .serial = true,
                .flushL2 = false}, 
};

/* static constexpr std::size_t N = 1e6; */
static constexpr host::Distribution weight_distribution = host::Distribution::SEEDED_RANDOM_UNIFORM;
static constexpr std::size_t min_SplitSize = 2;
static constexpr std::size_t max_SplitSize = 160;
static constexpr std::size_t step_splitSize = 1;
static constexpr std::size_t min_N = (1 << 16);
static constexpr std::size_t max_N = (1 << 28);
static constexpr std::size_t steps_N = 100;

static constexpr std::size_t iterations = 2;
static constexpr std::size_t flushSize = 1e7;

static constexpr MeanConfig meanConfig = AtomicMeanConfig();
static constexpr PrefixPartitionConfig prefixPartitionConfig = DecoupledPrefixPartitionConfig();

static constexpr bool writeElements = false;

struct ConfigResult {
    std::size_t N;
    std::size_t splitSize;
    std::size_t threadsPerPack;
    std::size_t workgroupSize;
    bool writeElements;
    double meanLatency;
    double meanLatency_std;
    double parScanLatency;
    double parScanLatency_std;
    double splitPackLatency;     // ms
    double splitPackLatency_std; // ms

    double splitLatency;
    double splitLatency_std;
    double packLatency;
    double packLatency_std;
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
                                       bool serial,
                                       bool flushL2) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
    assert(resourceExt != nullptr);
    auto alloc = resourceExt->resource_allocator();

    std::size_t threadsPerPack =
        context->physical_device.physical_device_subgroup_properties.subgroupSize /
        packsPerSubgroup;

    PhiloxBuffers flush =
        PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, flushSize);

    PSABuffers psaBuffers;
    if (serial) {
        if (threadsPerPack == 1) {
            psaBuffers = PSABuffers::allocate(
                alloc, merian::MemoryMappingType::NONE,
                PSAConfig(meanConfig, prefixPartitionConfig,
                          SerialSplitPackConfig(ScalarSplitConfig(min_SplitSize, workgroupSize),
                                                ScalarPackConfig(min_SplitSize, workgroupSize)),
                          writeElements),
                max_N);
        } else {
            psaBuffers = PSABuffers::allocate(
                alloc, merian::MemoryMappingType::NONE,
                PSAConfig(meanConfig, prefixPartitionConfig,
                          SerialSplitPackConfig(
                              ScalarSplitConfig(min_SplitSize, workgroupSize),
                              SubgroupPackConfig(min_SplitSize, packsPerSubgroup, workgroupSize)),
                          writeElements),
                max_N);
        }
    } else {
        psaBuffers = PSABuffers::allocate(
            alloc, merian::MemoryMappingType::NONE,
            PSAConfig(meanConfig, prefixPartitionConfig,
                      InlineSplitPackConfig(min_SplitSize, packsPerSubgroup, workgroupSize),
                      writeElements),
            max_N);
    }
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

    SplitPackBuffers splitPackBuffers;
    splitPackBuffers.heavyCount = psaBuffers.m_heavyCount;
    splitPackBuffers.mean = psaBuffers.m_mean;
    splitPackBuffers.partitionPrefix = psaBuffers.m_partitionPrefix;
    splitPackBuffers.partitionElements = psaBuffers.m_partitionElements;
    splitPackBuffers.partitionIndices = psaBuffers.m_partitionIndices;
    splitPackBuffers.weights = psaBuffers.weights;
    splitPackBuffers.aliasTable = psaBuffers.aliasTable;
    splitPackBuffers.m_internals = psaBuffers.m_splitPackInternals;

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

    std::size_t threadsPerSubproblem =
        context->physical_device.physical_device_subgroup_properties.subgroupSize /
        packsPerSubgroup;

    merian::ProfilerHandle benchmarkProfiler = std::make_shared<merian::Profiler>(context);

    for (std::size_t splitSize = min_SplitSize; splitSize <= max_SplitSize;
         splitSize += step_splitSize) {

        benchmarkProfiler->start("Split iteration");
        /* fmt::println("Progess : {}  ({} / {})", */
        /*              (splitSize - min_SplitSize) / (float)(max_SplitSize - min_SplitSize), */
        /*              splitSize, max_SplitSize); */

        if (threadsPerSubproblem >= splitSize) {
            continue;               // doesn't make any sense
        }

        benchmarkProfiler->start("Create pipeline");
        std::optional<SplitPackConfig> splitPackConfig = std::nullopt;
        if (serial) {
            if (threadsPerPack == 1) {
                splitPackConfig = SerialSplitPackConfig(ScalarSplitConfig(splitSize, workgroupSize),
                                                        ScalarPackConfig(splitSize, workgroupSize));
            } else {
                splitPackConfig = SerialSplitPackConfig(
                    ScalarSplitConfig(splitSize, workgroupSize),
                    SubgroupPackConfig(splitSize, packsPerSubgroup, workgroupSize));
            }
        } else {
            splitPackConfig = InlineSplitPackConfig(splitSize, packsPerSubgroup, workgroupSize);
        }

        SplitPack splitPack{context, shaderCompiler, splitPackConfig.value(), writeElements};
        benchmarkProfiler->end();

        benchmarkProfiler->start("Create measurement profiler");

        benchmarkProfiler->end();

        benchmarkProfiler->start("Sweep N");
        for (std::size_t n : host::exp::log10scale(min_N, max_N, steps_N)) {
            merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
            merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
                std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(
                    context, 16 * iterations);
            query_pool->reset();
            profiler->set_query_pool(query_pool);

            benchmarkProfiler->start("Generate input");
            { // Generate weights
                merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
                cmd->begin();
                prng.run(cmd, prngBuffers, n, dist(rng));
                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                                 vk::AccessFlagBits::eShaderRead));

                if (flushL2) {
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
                }

                cmd->end();
                queue->submit_wait(cmd);
            }
            benchmarkProfiler->end();

            std::string meanLabel = fmt::format("Mean: N={}", splitSize, n);
            std::string parScanLabel = fmt::format("PartitionScan: N={}", splitSize, n);
            std::string splitPackLabel = fmt::format("SplitPack-{}:N={}", splitSize, n);

            benchmarkProfiler->start("Iterations");
            merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
            cmd->begin();
            for (std::size_t i = 0; i < iterations; ++i) { // PSA

                {
                    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, meanLabel);
                    mean.run(cmd, meanBuffers, n);
                }

                cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                             vk::PipelineStageFlagBits::eComputeShader,
                             meanBuffers.mean->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead));

                {
                    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, parScanLabel);
                    prefixPartition.run(cmd, prefixPartitionBuffers, n);
                }

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
                    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, splitPackLabel);
                    splitPack.run(cmd, splitPackBuffers, n, profiler);
                }

                cmd->barrier(
                    vk::PipelineStageFlagBits::eAllCommands,
                    vk::PipelineStageFlagBits::eAllCommands,
                    {
                        prefixPartitionBuffers.partitionPrefix->buffer_barrier(
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                        prefixPartitionBuffers.partitionIndices->buffer_barrier(
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                        prefixPartitionBuffers.heavyCount->buffer_barrier(
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                        splitPackBuffers.aliasTable->buffer_barrier(
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                        prngBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderRead,
                                                            vk::AccessFlagBits::eShaderWrite),
                        flushBuffers.samples->buffer_barrier(vk::AccessFlagBits::eShaderRead,
                                                             vk::AccessFlagBits::eShaderWrite),
                    });
            }
            cmd->end();
            queue->submit_wait(cmd);
            profiler->collect(true, true);

            benchmarkProfiler->end();
            auto report = profiler->get_report();
            auto meanEntry = std::ranges::find_if(
                report.gpu_report, [&](const auto& entry) { return entry.name == meanLabel; });
            auto parScanEntry = std::ranges::find_if(
                report.gpu_report, [&](const auto& entry) { return entry.name == parScanLabel; });
            auto splitPackEntry = std::ranges::find_if(
                report.gpu_report, [&](const auto& entry) { return entry.name == splitPackLabel; });

            double packDur = -1;
            double packStd = -1;
            double splitDur = -1;
            double splitStd = -1;
            if (serial) {
                auto splitEntry =
                    std::ranges::find_if(splitPackEntry->children,
                                         [&](const auto& entry) { return entry.name == "Split"; });
                auto packEntry =
                    std::ranges::find_if(splitPackEntry->children,
                                         [&](const auto& entry) { return entry.name == "Pack"; });
                packDur = packEntry->duration;
                packStd = packEntry->std_deviation;

                splitDur = splitEntry->duration;
                splitStd = splitEntry->std_deviation;
            }

            results.entries.push_back(ConfigResult{
                .N = n,
                .splitSize = splitSize,
                .threadsPerPack = threadsPerPack,
                .workgroupSize = workgroupSize,
                .writeElements = false,
                .meanLatency = meanEntry->duration,
                .meanLatency_std = meanEntry->std_deviation,
                .parScanLatency = parScanEntry->duration,
                .parScanLatency_std = parScanEntry->std_deviation,
                .splitPackLatency = splitPackEntry->duration,
                .splitPackLatency_std = splitPackEntry->std_deviation,
                .splitLatency = splitDur,
                .splitLatency_std = splitStd,
                .packLatency = packDur,
                .packLatency_std = packStd,
            });
        }

        benchmarkProfiler->end();

        benchmarkProfiler->end();
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

    std::string path = "export/psa/splitpack/sweep.csv";
    host::exp::CSVWriter<18> csv(
        {"N", "splitSize", "method", "group", "meanLatency", "meanLatency_std", "parScanLatency",
         "parScanLatency_std", "splitPackLatency", "splitPackLatency_std", "flushL2",
         "workgroupSize", "threadsPerPack", "writeElements", "splitLatency", "splitLatency_std",
         "packLatency", "packLatency_std"},
        path);
    for (const auto& config : CONFIGURATIONS) {
        SPDLOG_INFO(
            "[{}%] Benchmarking {} ({})",
            (i / static_cast<float>(sizeof(CONFIGURATIONS) / (float)sizeof(CONFIGURATIONS[0]))) *
                100.0f,
            config.group, config.name);
        auto configBenchmark =
            benchmarkConfiguration(context, shaderCompiler, queue, config.packsPerSubgroup,
                                   config.workgroupSize, config.serial, config.flushL2);
        for (const auto& r2 : configBenchmark.entries) {
            csv.pushRow(r2.N, r2.splitSize, config.name, config.group, r2.meanLatency,
                        r2.meanLatency_std, r2.parScanLatency, r2.parScanLatency_std,
                        r2.splitPackLatency, r2.splitPackLatency_std, config.flushL2,
                        r2.workgroupSize, r2.threadsPerPack, r2.writeElements, r2.splitLatency,
                        r2.splitLatency_std, r2.packLatency, r2.packLatency_std);
        }
        csv.flush();
    }
    SPDLOG_INFO("Writing results to {}", path);
}

} // namespace device::psa_inline_splitpack_all
