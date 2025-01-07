#include "./psa.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/psa/PSA.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <cmath>
#include <fmt/base.h>
#include <string>

constexpr wrs::glsl::uint MEAN_WORKGROUP_SIZE = 512;
constexpr wrs::glsl::uint MEAN_ROWS = 4;
constexpr wrs::glsl::uint PREFIX_PARTITION_WORKGROUP_SIZE = 512;
constexpr wrs::glsl::uint PREFIX_PARTITION_ROWS = 4;
constexpr wrs::glsl::uint PREFIX_PARTITION_LOOKBACK_DEPTH = 32;
constexpr wrs::glsl::uint SPLIT_WORKGROUP_SIZE = 512;
constexpr wrs::glsl::uint SPLIT_SIZE = 32;
constexpr wrs::glsl::uint PACK_WORKGROUP_SIZE = 1;
constexpr wrs::glsl::uint SAMPLING_WORKGROUP_SIZE = 512;

/* constexpr wrs::glsl::uint COOPERATIVE_SAMPLING_SIZE = 4096; */

constexpr wrs::glsl::uint WEIGHT_COUNT = 1024 * 2048;
constexpr wrs::glsl::uint MIN_SAMPLES_COUNT = 1e3;
constexpr wrs::glsl::uint MAX_SAMPLES_COUNT = 1e7;
constexpr wrs::glsl::uint BENCHMARK_SAMPLES = 100;
constexpr wrs::glsl::uint BENCHMARK_ITERATIONS = 100;

void wrs::bench::psa::write_bench_results(const merian::ContextHandle& context) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 4096);
    query_pool->reset(); // LOL THIS WAS HARD TO FIND shared_ptr also defines a reset function =^).
    profiler->set_query_pool(query_pool);

    wrs::PSA psa{context, PSAConfig{
                              .psac =
                                  {
                                      .meanWorkgroupSize = MEAN_WORKGROUP_SIZE,
                                      .meanRows = MEAN_ROWS,
                                      .prefixSumWorkgroupSize = PREFIX_PARTITION_WORKGROUP_SIZE,
                                      .prefixSumRows = PREFIX_PARTITION_ROWS,
                                      .prefixSumLookbackDepth = PREFIX_PARTITION_LOOKBACK_DEPTH,
                                      .splitWorkgroupSize = SPLIT_WORKGROUP_SIZE,
                                      .packWorkgroupSize = PACK_WORKGROUP_SIZE,
                                      .splitSize = SPLIT_SIZE,
                                  },
                              .samplingWorkgroupSize = SAMPLING_WORKGROUP_SIZE,
                          }};

    using Buffers = wrs::PSA::Buffers;
    Buffers local =
        psa.allocate(alloc, merian::MemoryMappingType::NONE, WEIGHT_COUNT, MAX_SAMPLES_COUNT);
    Buffers stage = psa.allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, WEIGHT_COUNT,
                                 MAX_SAMPLES_COUNT);

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
                psa.run(cmd, local, WEIGHT_COUNT, S, profiler);
            }

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, {});
        }
        cmd.end();
        queue->submit_wait(cmd);
        profiler->collect(true, true);
    }

    profiler->collect(true, false);
    auto report = profiler->get_report();

    wrs::exp::CSVWriter<9> csv{
        {"sample_size", "latency", "sampling", "construction", "prepare", "mean", "prefix-partition", "split", "pack"},
        "./psa_benchmark.csv"};
    for (auto report : report.gpu_report) {
        float duration = report.duration;
        std::size_t S = std::stoi(report.name);
        float construction = NAN;
        float sampling = NAN;

        float prepare = NAN;
        float mean = NAN;
        float partition = NAN;
        float split = NAN;
        float pack = NAN;
        for (auto child : report.children) {
            if (child.name == "Construction") {
                construction = child.duration;
                for (auto step : child.children) {
                  if (step.name == "Prepare") {
                    prepare = step.duration;
                  }else if (step.name == "Mean") {
                    mean = step.duration;
                  } else if (step.name == "PrefixPartition") {
                    partition = step.duration;
                  } else if (step.name == "Split") {
                    split = step.duration;
                  } else if (step.name == "Pack") {
                    pack = step.duration;
                  }
                }
            } else if (child.name == "Sampling") {
                sampling = child.duration;
            }
        }
        csv.pushRow(S, duration, sampling, construction, prepare, mean, partition, split, pack);
    }

    /* fmt::println("{}",merian::Profiler::get_report_str(report)); */
}
