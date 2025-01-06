#include "./its.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/its/sampling/InverseTransformSampling.hpp"
#include <algorithm>
#include <ranges>
#include <ratio>

static void benchmark() {}

void wrs::bench::its::write_bench_results(const merian::ContextHandle& context) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
    query_pool->reset(); // LOL THIS WAS HARD TO FIND shared_ptr also defines a reset function =^).
    profiler->set_query_pool(query_pool);
  
    wrs::InverseTransformSampling its{context, 512};

    vk::CommandBuffer cmd = cmdPool->create_and_begin();

    cmd.end();
    queue->submit_wait(cmd);

    auto report = profiler->get_report();

    auto it = std::ranges::find_if(report.gpu_report, [](const merian::Profiler::ReportEntry& entry) { 
        return entry.name == "Target"; 
    });

    merian::Profiler::ReportEntry entry = *it;
    std::chrono::duration<double, std::milli> duration {entry.duration};


}
