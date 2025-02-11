#include "./partition.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/partition/Partition.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <algorithm>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

using namespace wrs;

constexpr std::array<PartitionConfig, 6> CONFIGURATIONS = {
    DecoupledPartitionConfig(512, // workgroup size
                             8,   // rows
                             BlockScanVariant::RANKED_STRIDED | BlockScanVariant::EXCLUSIVE |
                                 BlockScanVariant::SUBGROUP_SCAN_INTRINSIC,
                             32 // decoupled lookback depth
                             ),
    DecoupledPartitionConfig(512, // workgroup size
                             4,   // rows
                             BlockScanVariant::RANKED_STRIDED | BlockScanVariant::EXCLUSIVE |
                                 BlockScanVariant::SUBGROUP_SCAN_INTRINSIC,
                             32 // decoupled lookback depth
                             ),
    BlockWisePartitionConfig(
        PartitionBlockScanConfig(128, // workgroup size
                                 2,   // rows
                                 2,   // sequential scan length
                                 BlockScanVariant::RAKING | BlockScanVariant::EXCLUSIVE |
                                     BlockScanVariant::SUBGROUP_SCAN_INTRINSIC),
        BlockScanConfig(512, // workgroup size
                        2,   // rows
                        BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE |
                            BlockScanVariant::SUBGROUP_SCAN_INTRINSIC,
                        2, // sequential scan length
                        true),
        PartitionCombineConfig(256, // workgroup size
                               2,   // rows
                               1,   // sequential combine length
                               1    // blocks per workgroup
                               )),
    BlockWisePartitionConfig(
        PartitionBlockScanConfig(256, // workgroup size
                                 2,  // rows
                                 4,  // sequential scan length
                                 BlockScanVariant::RAKING | BlockScanVariant::EXCLUSIVE |
                                     BlockScanVariant::SUBGROUP_SCAN_INTRINSIC),
        BlockScanConfig(512, // workgroup size
                        2,   // rows
                        BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE |
                            BlockScanVariant::SUBGROUP_SCAN_INTRINSIC,
                        2, // sequential scan length
                        true),
        PartitionCombineConfig(512, // workgroup size
                               4,   // rows
                               1,   // sequential combine length
                               1    // blocks per workgroup
                               )),
    BlockWisePartitionConfig(
        PartitionBlockScanConfig(128,
                                 2,
                                 1,
                                 BlockScanVariant::RANKED_STRIDED | BlockScanVariant::EXCLUSIVE |
                                     BlockScanVariant::SUBGROUP_SCAN_INTRINSIC),
        BlockScanConfig(256,
                        2,
                        BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE |
                            BlockScanVariant::SUBGROUP_SCAN_INTRINSIC,
                        1,
                        true),
        PartitionCombineConfig(256, 1, 1, 1)),
    BlockWisePartitionConfig(
        PartitionBlockScanConfig(512,
                                 2,
                                 2,
                                 BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE |
                                     BlockScanVariant::SUBGROUP_SCAN_INTRINSIC),
        BlockScanConfig(512,
                        2,
                        BlockScanVariant::RANKED | BlockScanVariant::EXCLUSIVE |
                            BlockScanVariant::SUBGROUP_SCAN_SHFL,
                        2,
                        true),
        PartitionCombineConfig(512, 4, 1, 1)),

};

constexpr std::size_t MAX_N = (1 << 24);
constexpr std::size_t MIN_N = (1 << 12);
constexpr std::size_t ITERATIONS = 100;
constexpr std::size_t TICKS = 100;
/* static_assert(ITERATIONS * TICKS < 1024); // Avoids bug in merian::Profiler */

using Base = float; // only supported base type right now!

struct BenchmarkResult {
    std::vector<std::optional<float>> durations; // in ms.
};
BenchmarkResult benchmarkConfiguration(const merian::ContextHandle& context,
                                       const merian::ResourceAllocatorHandle& alloc,
                                       const merian::QueueHandle& queue,
                                       const merian::CommandPoolHandle& cmdPool,
                                       wrs::eval::log10::IntLogScaleRange<glsl::uint> elementCounts,
                                       std::size_t maxN,
                                       PartitionConfig config) {

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, ITERATIONS * TICKS);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    std::vector<std::optional<float>> durations(elementCounts.size());

    using Buffers = Partition<Base>::Buffers;
    Buffers stage =
        Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, config, maxN);
    assert(stage.elements != nullptr);
    Buffers local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, config, maxN);
    assert(local.elements != nullptr);

    Partition<Base> method{context, config};
    // Upload some input date
    {
        SPDLOG_INFO("Uploading mock input for benchmarking");
        SPDLOG_DEBUG("Generating input");
        const std::vector<Base> weights =
            wrs::generate_weights<Base>(Distribution::PSEUDO_RANDOM_UNIFORM, maxN);

        vk::CommandBuffer cmd = cmdPool->create_and_begin();
        {
            SPDLOG_DEBUG("Upload elements");
            Buffers::ElementsView<Base> stageView{stage.elements, maxN};
            Buffers::ElementsView<Base> localView{local.elements, maxN};
            stageView.upload<float>(weights);
            stageView.copyTo(cmd, localView);
            localView.expectComputeRead(cmd);
        }
        {
            SPDLOG_DEBUG("Upload pivot = 0.5");
            Buffers::PivotView<Base> stageView{stage.pivot};
            Buffers::PivotView<Base> localView{local.pivot};
            stageView.upload<glsl::f32>(0.5);
            stageView.copyTo(cmd, localView);
            localView.expectComputeRead(cmd);
        }
        cmd.end();
        SPDLOG_DEBUG("Waiting for uploading to complete");
        queue->submit_wait(cmd);
    }

    // Benchmarking
    std::string prefix = partitionConfigName(config);
    SPDLOG_INFO("Begin benchmarking {}", prefix);
    std::size_t x = 0;
    for (const auto& n : elementCounts) {
        if (n <= method.maxElementCount()) {
            vk::CommandBuffer cmd = cmdPool->create_and_begin();
            std::string label = fmt::format("{}-{}", prefix, n);
            for (std::size_t it = 0; it < ITERATIONS; ++it) {
                profiler->start(label);
                profiler->cmd_start(cmd, label);

                method.run(cmd, local, n);

                profiler->end();
                profiler->cmd_end(cmd);
            }
            cmd.end();
            queue->submit_wait(cmd);

            profiler->collect(true, true);
            auto report = profiler->get_report();
            const auto it = std::ranges::find_if(report.gpu_report, [&](const auto& reportEntry) {
                return reportEntry.name == label;
            });
            assert(it != report.gpu_report.end());
            auto reportEntry = *it;
            durations[x] = reportEntry.duration;
            SPDLOG_DEBUG("Benchmark result: n={} took {}ms   variance={}", n, reportEntry.duration,
                         reportEntry.std_deviation);
        } else {
            durations[x] = std::nullopt;
        }

        x++;
    }
    SPDLOG_INFO("End benchmarking configuration");

    return BenchmarkResult{
        .durations = std::move(durations),
    };
}

void wrs::bench::partition::write_bench_results(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    auto elementCounts = wrs::eval::log10scale<glsl::uint>(MIN_N, MAX_N, TICKS);

    std::array<std::string, CONFIGURATIONS.size() + 1> benchmarkNames;
    benchmarkNames[0] = "element_count";
    std::array<BenchmarkResult, CONFIGURATIONS.size()> results;

    for (std::size_t i = 0; i < CONFIGURATIONS.size(); ++i) {

        results[i] = benchmarkConfiguration(context, alloc, queue, cmdPool, elementCounts, MAX_N,
                                            CONFIGURATIONS[i]);
        benchmarkNames[i + 1] = partitionConfigName(CONFIGURATIONS[i]);
    }

    const std::string filePath = "./partition_benchmark.csv";
    wrs::exp::CSVWriter<CONFIGURATIONS.size() + 1> csv{benchmarkNames, filePath};

    std::size_t x = 0;
    for (const auto& n : elementCounts) {
        csv.unsafePushValue(n, false);
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i].durations[x].has_value()) {
                csv.unsafePushValue(results[i].durations[x].value(), i == results.size() - 1);
            } else {
                csv.unsafePushNull(i == results.size() - 1);
            }
        }
        csv.unsafeEndRow();
        x++;
    }
}
