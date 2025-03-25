#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include <memory_resource>
namespace host::test {

enum TestResultType : uint8_t {
    SUCCESS = 0,
    WARNING = 1,
    ERROR = 2,
};

inline TestResultType operator+(TestResultType lhs, TestResultType rhs) {
    return static_cast<TestResultType>(
        std::max(static_cast<uint8_t>(lhs), static_cast<uint8_t>(rhs)));
}

inline TestResultType& operator+=(TestResultType& lhs, TestResultType rhs) {
    return lhs = static_cast<TestResultType>(
               std::max(static_cast<uint8_t>(lhs), static_cast<uint8_t>(rhs)));
}

struct TestProperty {
    std::string name;
    std::string value;
};

struct TestContext {
    merian::ContextHandle context;
    merian::ResourceAllocatorHandle alloc;
    merian::QueueHandle queue;
    [[deprecated("Global command buffer leads to resources not beeing freed")]] merian::CommandPoolHandle cmdPool;
    merian::ShaderCompilerHandle shaderCompiler;
    // only setup if we benchmark incredibly large datasets and have to avoid memory fragmentation
    // Currently the default allocator
    std::pmr::memory_resource* memory_resource;

    void resetMemoryResource() const;

    void pushResult(const std::string& suit,
                    const std::string& test,
                    TestResultType result,
                    double duration,
                    double std_derivation,
                    std::initializer_list<TestProperty> properties = {}) const;

    merian::ResourceAllocatorHandle createResourceAllocator() const {
        auto resources = context->get_extension<merian::ExtensionResources>();
        merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
        return alloc;
    }

    void printPrettyLog() const;
};

TestContext setupTestContext(const merian::ContextHandle& context);
} // namespace host::test
