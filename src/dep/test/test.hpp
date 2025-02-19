#pragma once

#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/utils/profiler.hpp"

namespace wrs::test {

struct TestContext {
    merian::ContextHandle context;
    merian::ResourceAllocatorHandle alloc;
    merian::QueueHandle queue;
    merian::CommandPoolHandle cmdPool;
    merian::ProfilerHandle profiler;
    merian::ShaderCompilerHandle shaderCompiler;
};

TestContext setupTestContext(const merian::ContextHandle& context);

void testTests();

}; // namespace wrs::test


