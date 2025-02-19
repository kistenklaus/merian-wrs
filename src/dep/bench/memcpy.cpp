#include "./memcpy.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/types/glsl.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

using namespace wrs;

constexpr std::size_t MIN_N = (1 << 12);
constexpr std::size_t MAX_N = (1 << 24);
constexpr std::size_t TICKS = 100;

constexpr std::size_t ITERATIONS = 100;

void wrs::bench::memcpy::write_bench_results(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, ITERATIONS * TICKS);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    auto elementCounts = wrs::eval::log10scale<glsl::uint>(MIN_N, MAX_N, TICKS);

    const std::string filePath = "./memcpy_benchmark.csv";

    SPDLOG_DEBUG("Writing results to {}", filePath);
    wrs::exp::CSVWriter<2> csv{{"element_count", "memcpy"}, filePath};

    merian::BufferHandle src;
    merian::BufferHandle dst;

    {
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();

        merian::BufferHandle stage =
            alloc->createBuffer(MAX_N * sizeof(float), vk::BufferUsageFlagBits::eTransferSrc,
                                merian::MemoryMappingType::HOST_ACCESS_RANDOM);

        src = alloc->createBuffer(MAX_N * sizeof(float),
                                  vk::BufferUsageFlagBits::eTransferDst |
                                      vk::BufferUsageFlagBits::eTransferSrc,
                                  merian::MemoryMappingType::NONE);

        dst = alloc->createBuffer(MAX_N * sizeof(float),
                                  vk::BufferUsageFlagBits::eTransferDst |
                                      vk::BufferUsageFlagBits::eTransferSrc,
                                  merian::MemoryMappingType::NONE);

        std::vector<float> mock = wrs::generate_weights(Distribution::PSEUDO_RANDOM_UNIFORM, MAX_N);
        void* x = stage->get_memory()->map();
        std::memcpy(x, mock.data(), mock.size() * sizeof(float));
        stage->get_memory()->unmap();

        vk::BufferCopy copy{0, 0, MAX_N * sizeof(float)};
        cmd->copy(stage, src, copy);
        cmd->end();
        queue->submit_wait(cmd);
    }

    for (const auto& n : elementCounts) {

        double min = 1000000;
        for (std::size_t x = 0; x < ITERATIONS; ++x) {
            std::string label = fmt::format("{}", n);
            merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
            cmd->begin();
            for (std::size_t i = 0; i < 10; ++i) {
                vk::DeviceSize offset = 0;
                vk::BufferCopy copy{offset * sizeof(float), offset * sizeof(float),
                                    n * sizeof(float)};
                {
                    profiler->start(label);
                    profiler->cmd_start(cmd, label);
                    cmd->copy(src, dst, copy);
                    profiler->end();
                    profiler->cmd_end(cmd);
                }

                std::swap(src, dst);
            }
            cmd->end();
            queue->submit_wait(cmd);
            profiler->collect(true, true);
            auto report = profiler->get_report();
            const auto it = std::ranges::find_if(report.gpu_report, [&](const auto& reportEntry) {
                return reportEntry.name == label;
            });
            assert(it != report.gpu_report.end());
            auto reportEntry = *it;
            min = std::min(reportEntry.duration, min);
        }
        double duration = min;
        SPDLOG_INFO("N = {} took {}ms", n, duration);
        csv.pushRow(n, duration);
    }
}
