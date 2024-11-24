#include "./test.hpp"

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
/* #include "./compare_ranges.hpp" */
#include "./is_prefix.hpp"



wrs::test::TestContext wrs::test::setupTestContext(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
    query_pool.reset();
    profiler->set_query_pool(query_pool);

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    std::vector<float> a = {1,2,3};
    std::vector<float> b = {1,3,6};
    assert_is_inclusive_prefix<float>(std::span(a), std::span(b));
    /* compare_ranges(a, b); */

    return {
        .alloc = alloc,
        .queue = queue,
        .cmdPool = cmdPool,
        .profiler = profiler,
        .__profilerQueryPool = query_pool,
    };
}


