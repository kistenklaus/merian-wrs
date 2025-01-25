#include "./hst.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/hs/HS.hpp"
#include "src/wrs/algorithm/hs/HSTRepr.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/types/glsl.hpp"
#include <algorithm>

using namespace wrs;
constexpr glsl::uint N = 1024 * 2048;
constexpr Distribution DIST = Distribution::UNIFORM;
constexpr glsl::uint S = 1e8;

constexpr glsl::uint HSTC_WORKGROUP_SIZE = 512;
constexpr glsl::uint SVO_WORKGROUP_SIZE = 512;
constexpr glsl::uint SAMPLING_WORKGROUP_SIZE = 512;
constexpr glsl::uint EXPLODE_WORKGROUP_SIZE = 512;
constexpr glsl::uint EXPLODE_ROWS = 8;
constexpr glsl::uint EXPLODE_LOOKBACK_DEPTH = 32;

constexpr glsl::uint BENCHMARK_TICKS = 100;
constexpr glsl::uint BENCHMARK_ITERATIONS = 10;

struct BenchmarkResults {
    float latencyMs;
    float prepareMs;
    float constructionMs;
    float svoMs;
    float samplingMs;
    float explodeMs;
};

BenchmarkResults benchmark(const merian::CommandPoolHandle& cmdPool,
                           const merian::QueueHandle& queue,
                           const merian::ProfilerHandle& profiler,
                           const wrs::HS& hs,
                           const wrs::HS::Buffers& local,
                           const wrs::HS::Buffers& stage,
                           std::size_t s,
                           std::size_t N) {
    vk::CommandBuffer cmd = cmdPool->create_and_begin();
    wrs::hst::HSTRepr repr{N};

    glsl::uint* mapped = stage.outputSensitiveSamples->get_memory()->map_as<glsl::uint>();
    mapped[repr.size()] = s;
    stage.outputSensitiveSamples->get_memory()->unmap();
    vk::BufferCopy copy{
        repr.size() * sizeof(glsl::uint),
        repr.size() * sizeof(glsl::uint),
        sizeof(glsl::uint),
    };
    cmd.copyBuffer(*stage.outputSensitiveSamples, *local.outputSensitiveSamples, 1, &copy);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        local.outputSensitiveSamples->buffer_barrier(
                            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                        {});

    for (std::size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Latency");
        hs.run(cmd, local, N, s, profiler);
    }

    cmd.end();
    queue->submit_wait(cmd);

    profiler->collect(true, false);
    auto report = profiler->get_report();
    BenchmarkResults results;
    auto it = std::ranges::find_if(report.gpu_report,
                                   [](const auto& gpu) { return gpu.name == "Latency"; });
    results.latencyMs = it->duration;
    for (const auto& gpu : it->children) {
        if (gpu.name == "Prepare") {
            results.prepareMs = gpu.duration;
        } else if (gpu.name == "Construction") {
            results.constructionMs = gpu.duration;
        } else if (gpu.name == "Sampling") {
            results.samplingMs = gpu.duration;
        } else if (gpu.name == "Explode") {
            results.explodeMs = gpu.duration;
        } else if (gpu.name == "SVO") {
            results.svoMs = gpu.duration;
        }
    }

    return results;
}

void wrs::bench::hst::write_bench_results(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 4096);
    query_pool->reset(); // LOL THIS WAS HARD TO FIND shared_ptr also defines a reset function =^).
    profiler->set_query_pool(query_pool);

    const std::vector<float> weights = wrs::generate_weights(DIST, N);

    wrs::HS::Buffers local = wrs::HS::Buffers::allocate(alloc, merian::MemoryMappingType::NONE, N,
                                                        S, EXPLODE_WORKGROUP_SIZE * EXPLODE_ROWS);

    wrs::HS::Buffers stage =
        wrs::HS::Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, N, S,
                                   EXPLODE_WORKGROUP_SIZE * EXPLODE_ROWS);

    vk::CommandBuffer cmd = cmdPool->create_and_begin();

    wrs::HS::Buffers::WeightTreeView stageView{stage.weightTree, N};
    wrs::HS::Buffers::WeightTreeView localView{local.weightTree, N};
    stageView.upload<float>(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);

    cmd.end();
    queue->submit_wait(cmd);

    wrs::HS hs{context,
               HSTC_WORKGROUP_SIZE,
               SVO_WORKGROUP_SIZE,
               SAMPLING_WORKGROUP_SIZE,
               EXPLODE_WORKGROUP_SIZE,
               EXPLODE_ROWS,
               EXPLODE_LOOKBACK_DEPTH};

    wrs::exp::CSVWriter<7> csv{
        {"sample_size", "latency_ms", "prepare_ms", "construction_ms", "svo_ms", "sampling_ms", "explode_ms"},
        "hst_benchmark.csv"};

    for (const auto& s : wrs::eval::log10scale<glsl::uint>(1000, S, BENCHMARK_TICKS)) {
        auto result = benchmark(cmdPool, queue, profiler, hs, local, stage, s, N);

        csv.pushRow(s, result.latencyMs, result.prepareMs, result.constructionMs, result.svoMs, result.samplingMs,
                    result.explodeMs);
    }
}
