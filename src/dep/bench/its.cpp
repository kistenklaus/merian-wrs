#include "./its.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler_shaderc.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/its/ITS.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <cmath>
#include <fmt/base.h>
#include <string>

constexpr wrs::ITSConfig config = {};

constexpr wrs::glsl::uint WEIGHT_COUNT = 1024 * 2048;
constexpr wrs::glsl::uint MIN_SAMPLES_COUNT = 1e3;
constexpr wrs::glsl::uint MAX_SAMPLES_COUNT = 1e8;
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

    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);

    wrs::ITS its{context, shaderCompiler, config};

    using Buffers = wrs::ITS::Buffers;
    Buffers local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, WEIGHT_COUNT,
                                      MAX_SAMPLES_COUNT, config.prefixSumConfig.partitionSize());
    Buffers stage =
        Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, WEIGHT_COUNT,
                          MAX_SAMPLES_COUNT, config.prefixSumConfig.partitionSize());

    std::vector<float> weights =
        wrs::generate_weights(Distribution::SEEDED_RANDOM_UNIFORM, WEIGHT_COUNT);

    merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
    cmd->begin();
    { // Upload weights
        Buffers::WeightsView stageView{stage.weights, WEIGHT_COUNT};
        Buffers::WeightsView localView{local.weights, WEIGHT_COUNT};
        stageView.upload<float>(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }

    cmd->end();
    queue->submit_wait(cmd);

    auto scale =
        wrs::eval::log10scale<glsl::uint>(MIN_SAMPLES_COUNT, MAX_SAMPLES_COUNT, BENCHMARK_SAMPLES);
    for (const glsl::uint S : scale) {
        fmt::println("Benchmarking S = {}", S);
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();
        for (std::size_t it = 0; it < BENCHMARK_ITERATIONS; ++it) {
            {
                MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("{}", S));
                its.run(cmd, local, WEIGHT_COUNT, S, profiler);
            }

            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eComputeShader,
                         local.prefixSum->buffer_barrier(vk::AccessFlagBits::eHostRead,
                                                         vk::AccessFlagBits::eHostWrite));
        }
        cmd->end();
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
