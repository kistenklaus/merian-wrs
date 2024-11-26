#pragma once

#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include <memory>
#include <tuple>
namespace wrs::test {

struct TestContext {
    merian::ContextHandle context;
    merian::ResourceAllocatorHandle alloc;
    merian::QueueHandle queue;
    merian::CommandPoolHandle cmdPool;
    merian::ProfilerHandle profiler;
    /* merian::QueryPoolHandle<vk::QueryType::eTimestamp> __profilerQueryPool; */
};

TestContext setupTestContext(const merian::ContextHandle& context);

void testTests();

}; // namespace wrs::test
