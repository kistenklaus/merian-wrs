#include "src/device/memcpy/Memcpy.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prng/PRNG.hpp"
#include "src/device/prng/philox/Philox.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <random>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

namespace device::memcpy {

using weight_type = float;
using Buffers = Memcpy<weight_type>::Buffers;

enum Variant {
    MEMCPY_VK_API_COPY,
    MEMCPY_COMPUTE,
};

struct NamedConfig {
    std::string name;
    Variant variant;
    std::size_t ROWS;
    bool flushL2;
};

static const NamedConfig CONFIGURATIONS[] = {
    NamedConfig{
        .name = "Memcpy-FlushL2",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 16,
        .flushL2 = true,
    },
    NamedConfig{
        .name = "Memcpy-FlushL2",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 32,
        .flushL2 = true,
    },
    NamedConfig{
        .name = "Memcpy-FlushL2",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 512,
        .flushL2 = true,
    },
    NamedConfig{
        .name = "Memcpy-FlushL2",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 8,
        .flushL2 = true,
    },
    NamedConfig{.name = "Memcpy-FlushL2", .variant = MEMCPY_COMPUTE, .ROWS = 4, .flushL2 = true},
    NamedConfig{
        .name = "Memcpy-FlushL2",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 2,
        .flushL2 = true,
    },
    NamedConfig{
        .name = "Memcpy",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 16,
        .flushL2 = false,
    },
    NamedConfig{
        .name = "Memcpy",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 32,
        .flushL2 = false,
    },
    NamedConfig{
        .name = "Memcpy",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 512,
        .flushL2 = false,
    },
    NamedConfig{
        .name = "Memcpy",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 8,
        .flushL2 = false,
    },
    NamedConfig{
        .name = "Memcpy",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 4,
        .flushL2 = false,
    },
    NamedConfig{
        .name = "Memcpy",
        .variant = MEMCPY_COMPUTE,
        .ROWS = 2,
        .flushL2 = false,
    },
    NamedConfig{
        .name = "vkCmdCopyBuffer",
        .variant = MEMCPY_VK_API_COPY,
        .ROWS = 4,
        .flushL2 = false,
    },
    NamedConfig{
        .name = "vkCmdCopyBuffer-FlushL2",
        .variant = MEMCPY_VK_API_COPY,
        .ROWS = 4,
        .flushL2 = true,
    },
};

static constexpr std::size_t N = (1 << 28);
static constexpr std::size_t N_min = (1 << 16);
static constexpr std::size_t ticks = 500;
static constexpr std::size_t iterations = 10;

struct ConfigResult {
    std::size_t N;
    std::size_t bytes;
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
                                       const Variant variant,
                                       std::size_t rows,
                                       bool flushL2) {
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    Buffers local;
    Buffers stage;
    PhiloxBuffers temp;
    { // Setup
        const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
        assert(resourceExt != nullptr);
        auto alloc = resourceExt->resource_allocator();

        stage = Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, N);

        local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, N);

        temp = PhiloxBuffers::allocate(alloc, merian::MemoryMappingType::NONE, N);
    }
    ConfigBenchmark results;
    results.entries.reserve(ticks);

    PRNG prng{context, shaderCompiler, PhiloxConfig(512)};
    PRNGBuffers prngBuffers;
    prngBuffers.samples = local.src;
    PRNGBuffers flushBuffers;
    flushBuffers.samples = temp.samples;

    std::mt19937 rng;
    std::uniform_int_distribution<host::glsl::uint> dist;

    if (variant == MEMCPY_COMPUTE) {
        Memcpy<weight_type> memcpy{context, shaderCompiler, MemcpyConfig(512, rows)};

        for (const std::size_t n : host::exp::log10scale<std::size_t>(N_min, N, ticks)) {

            SPDLOG_INFO("N = {}", n);

            merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
            merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
                std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context,
                                                                               4 * iterations);
            query_pool->reset();
            profiler->set_query_pool(query_pool);

            for (std::size_t i = 0; i < iterations; ++i) {
                {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();

                    prng.run(cmd, prngBuffers, n, dist(rng));

                    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 local.src->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                           vk::AccessFlagBits::eShaderWrite));
                    cmd->end();
                    queue->submit_wait(cmd);
                }
                if (flushL2) {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();

                    prng.run(cmd, flushBuffers, N, dist(rng));

                    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 local.src->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                           vk::AccessFlagBits::eShaderWrite));

                    cmd->end();
                    queue->submit_wait(cmd);
                }

                {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();
                    /* profiler->start("Build"); */
                    /* profiler->cmd_start(cmd, "Build", vk::PipelineStageFlagBits::eTopOfPipe); */
                    memcpy.run(cmd, local, n, profiler);
                    /* profiler->end(); */
                    /* profiler->cmd_end(cmd, vk::PipelineStageFlagBits::eBottomOfPipe); */
                    cmd->end();
                    queue->submit_wait(cmd);
                }

                if (flushL2) {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();

                    prng.run(cmd, flushBuffers, N, dist(rng));

                    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 local.src->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                           vk::AccessFlagBits::eShaderWrite));

                    cmd->end();
                    queue->submit_wait(cmd);
                }
            }

            profiler->collect(true, true);
            fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report()));

            auto report = profiler->get_report();
            auto entry = std::ranges::find_if(
                report.gpu_report, [](const auto& entry) { return entry.name == "Memcpy"; });

            double latency = entry->duration;
            double stdVar = entry->std_deviation;

            std::size_t requireTransactionsByte = 2 * n * sizeof(weight_type); // in bytes
            double memoryBandwidth = ((requireTransactionsByte * 1e-9f) / (entry->duration * 1e-3));
            double itemsPerSecond = (n / (entry->duration * 1e-3)) / 1e9;

            SPDLOG_DEBUG("memory-throughput: {}Gb/s", memoryBandwidth);
            SPDLOG_DEBUG("utilization: {}%", (memoryBandwidth / 504.20f) * 100);

            results.entries.push_back(ConfigResult{.N = n,
                                                   .bytes = n * sizeof(float),
                                                   .latency = latency,
                                                   .stdVar = stdVar,
                                                   .memoryBandwidth = memoryBandwidth,
                                                   .throughput = itemsPerSecond});
        }

    } else {

        for (const std::size_t n : host::exp::log10scale<std::size_t>(N_min, N, ticks)) {
            queue->wait_idle();

            SPDLOG_INFO("N = {}", n);

            merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
            merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
                std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context,
                                                                               4 * iterations);
            query_pool->reset();
            profiler->set_query_pool(query_pool);

            for (std::size_t i = 0; i < iterations; ++i) {
                {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();

                    prng.run(cmd, prngBuffers, n, dist(rng));

                    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 local.src->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                           vk::AccessFlagBits::eShaderWrite));
                    cmd->end();
                    queue->submit_wait(cmd);
                }
                if (flushL2) {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();

                    prng.run(cmd, flushBuffers, N, dist(rng));

                    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 local.src->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                           vk::AccessFlagBits::eShaderWrite));

                    cmd->end();
                    queue->submit_wait(cmd);
                }
                {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();
                    profiler->start("Build");
                    profiler->cmd_start(cmd, "Build");

                    vk::BufferCopy copy{0, 0, n * sizeof(float)};
                    cmd->copy(local.dst, local.src, copy);

                    profiler->end();
                    profiler->cmd_end(cmd);

                    cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eTransfer,
                                 local.dst->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                           vk::AccessFlagBits::eTransferWrite));
                    cmd->end();
                    queue->submit_wait(cmd);
                }

                if (flushL2) {
                    merian::CommandBufferHandle cmd =
                        std::make_shared<merian::CommandBuffer>(cmdPool);
                    cmd->begin();

                    prng.run(cmd, flushBuffers, N, dist(rng));

                    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 local.src->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                           vk::AccessFlagBits::eShaderWrite));

                    cmd->end();
                    queue->submit_wait(cmd);
                }
            }

            profiler->collect(true, true);

            auto report = profiler->get_report();
            auto entry = std::ranges::find_if(
                report.gpu_report, [](const auto& entry) { return entry.name == "Build"; });

            double latency = entry->duration;
            double stdVar = entry->std_deviation;

            std::size_t requireTransactionsByte = n * sizeof(weight_type);
            double memoryBandwidth = ((requireTransactionsByte * 1e-9) / (entry->duration * 1e-3));
            double itemsPerSecond = (n / (entry->duration * 1e-3)) / 1e9;

            results.entries.push_back(ConfigResult{.N = n,
                                                   .bytes = n * sizeof(float),
                                                   .latency = latency,
                                                   .stdVar = stdVar,
                                                   .memoryBandwidth = memoryBandwidth,
                                                   .throughput = itemsPerSecond});
        }
    }
    return results;
}

void benchmark(const merian::ContextHandle& context) {
    // Setup vulkan resources
    merian::QueueHandle queue = context->get_queue_GCT();

    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);

    BenchmarkResults results;
    for (const auto& config : CONFIGURATIONS) {
        auto configBenchmark = benchmarkConfiguration(context, shaderCompiler, queue,
                                                      config.variant, config.ROWS, config.flushL2);
        results.entries.push_back(BenchmarkResult{
            .configuration = config,
            .results = configBenchmark,
        });
    }

    // export

    std::string path = "memcpy_benchmark.csv";
    host::exp::CSVWriter<7> csv(
        {"N", "bytes", "method", "latency", "std_derivation", "throughput", "memory_throughput"},
        path);
    for (const auto& r1 : results.entries) {
        std::string method = r1.configuration.name;
        for (const auto& r2 : r1.results.entries) {
            csv.pushRow(r2.N, r2.bytes, method, r2.latency, r2.stdVar, r2.throughput,
                        r2.memoryBandwidth);
        }
    }
}

} // namespace device::memcpy
