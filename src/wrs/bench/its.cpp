#include "./its.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/its/ITS.hpp"
#include "src/wrs/algorithm/its/sampling/InverseTransformSampling.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/layout/BufferView.hpp"
#include <algorithm>
#include <cmath>
#include <fmt/base.h>
#include <ratio>
#include <string>

constexpr wrs::glsl::uint PREFIX_SUM_WORKGROUP_SIZE = 512;
constexpr wrs::glsl::uint PREFIX_SUM_ROWS = 8;
constexpr wrs::glsl::uint PREFIX_SUM_PARALLEL_LOOKBACK_DEPTH = 32;
constexpr wrs::glsl::uint SAMPLING_WORKGROUP_SIZE = 512;
constexpr wrs::glsl::uint COOPERATIVE_SAMPLING_SIZE = 4096;

constexpr wrs::glsl::uint WEIGHT_COUNT = 1024 * 2048;
constexpr wrs::glsl::uint MIN_SAMPLES_COUNT = 1e3;
constexpr wrs::glsl::uint MAX_SAMPLES_COUNT = 1e7;
constexpr wrs::glsl::uint BENCHMARK_SAMPLES = 500;
constexpr wrs::glsl::uint BENCHMARK_ITERATIONS = 10;

void wrs::bench::its::write_bench_results(const merian::ContextHandle& context) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 4096);
    query_pool->reset(); // LOL THIS WAS HARD TO FIND shared_ptr also defines a reset function =^).
    profiler->set_query_pool(query_pool);

    wrs::ITS its{context,
                 PREFIX_SUM_WORKGROUP_SIZE,
                 PREFIX_SUM_ROWS,
                 PREFIX_SUM_PARALLEL_LOOKBACK_DEPTH,
                 SAMPLING_WORKGROUP_SIZE,
                 COOPERATIVE_SAMPLING_SIZE};

    using Buffers = wrs::ITS::Buffers;
    Buffers local =
        Buffers::allocate(alloc, merian::MemoryMappingType::NONE, WEIGHT_COUNT, MAX_SAMPLES_COUNT,
                          PREFIX_SUM_WORKGROUP_SIZE * PREFIX_SUM_ROWS);
    Buffers stage =
        Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, WEIGHT_COUNT,
                          MAX_SAMPLES_COUNT, PREFIX_SUM_WORKGROUP_SIZE * PREFIX_SUM_ROWS);

    std::vector<float> weights =
        wrs::generate_weights(Distribution::SEEDED_RANDOM_UNIFORM, WEIGHT_COUNT);

    vk::CommandBuffer cmd = cmdPool->create_and_begin();
    { // Upload weights
        Buffers::WeightsView stageView{stage.weights, WEIGHT_COUNT};
        Buffers::WeightsView localView{local.weights, WEIGHT_COUNT};
        stageView.upload<float>(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }

    cmd.end();
    queue->submit_wait(cmd);

    auto scale =
        wrs::eval::log10scale<glsl::uint>(MIN_SAMPLES_COUNT, MAX_SAMPLES_COUNT, BENCHMARK_SAMPLES);
    for (const glsl::uint S : scale) {
        fmt::println("Benchmarking S = {}", S);
        vk::CommandBuffer cmd = cmdPool->create_and_begin();
        for (std::size_t it = 0; it < BENCHMARK_ITERATIONS; ++it) {
            {
                MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("{}", S));
                its.run(cmd, local, WEIGHT_COUNT, S, profiler);
            }

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader, {}, {},
                                local.prefixSum->buffer_barrier(vk::AccessFlagBits::eHostRead,
                                                                vk::AccessFlagBits::eHostWrite),
                                {});
        }
        cmd.end();
        queue->submit_wait(cmd);
        profiler->collect(true, true);
    }

    profiler->collect(true, false);
    auto report = profiler->get_report();

    wrs::exp::CSVWriter<5> csv{
        {"sample_size", "latency_ms", "prepare_ms", "prefix_sum_ms", "sampling_ms"},
        "./its_benchmark.csv"};
    for (auto report : report.gpu_report) {
        float duration = report.duration;
        std::size_t S = std::stoi(report.name);
        float prepare = NAN;
        float prefixSum = NAN;
        float sampling = NAN;
        for (auto child : report.children) {
            if (child.name == "Prepare") {
                prepare = child.duration;
            } else if (child.name == "Prefix Sum") {
                prefixSum = child.duration;
            } else if (child.name == "Sampling") {
                sampling = child.duration;
            }
        }
        csv.pushRow(S, duration, prepare, prefixSum, sampling);
    }

    /* fmt::println("{}",merian::Profiler::get_report_str(report)); */
}
