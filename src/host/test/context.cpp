#include "./context.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <memory_resource>
#include <ranges>

struct TestResult {
    std::string suitName;
    std::string testName;
    host::test::TestResultType type;
    double duration;
    double std_derivation;
    std::vector<host::test::TestProperty> properties;
};

static std::vector<TestResult> g_testSuitResults;

host::test::TestContext host::test::setupTestContext(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();

    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    return {
        .context = context,
        .alloc = alloc,
        .queue = queue,
        .cmdPool = cmdPool,
        .shaderCompiler = shaderCompiler,
        .memory_resource = std::pmr::get_default_resource(), // pretty useless idea.
    };
}
void host::test::TestContext::resetMemoryResource() const {}

void host::test::TestContext::pushResult(
    const std::string& suit,
    const std::string& test,
    host::test::TestResultType result,
    double duration,
    double std_derivation,
    std::initializer_list<host::test::TestProperty> properties) const {
    g_testSuitResults.push_back(TestResult{
        .suitName = suit,
        .testName = test,
        .type = result,
        .duration = duration,
        .std_derivation = std_derivation,
        .properties = properties,
    });
}
void host::test::TestContext::printPrettyLog() const {
    // Oraganize results

    std::vector<std::string> suites;
    for (const auto& result : g_testSuitResults) {
        if (std::ranges::find(suites, result.suitName) == suites.end()) {
            suites.push_back(result.suitName);
        }
    }

    // Pretty printing

    for (const auto& suitName : suites) {
        fmt::println("{:=^120}", suitName);
        auto tests = g_testSuitResults | std::views::filter([&suitName](const auto& test) {
                         return test.suitName == suitName;
                     });
        std::size_t maxTestNameLength = std::ranges::max(
            tests | std::views::transform([](const auto& t) { return t.testName.size(); }));
        for (const auto& test : tests) {
            switch (test.type) {
            case SUCCESS:
                fmt::print("\x1B[32mSUCCESS\x1B[0m");
                break;
            case WARNING:
                fmt::print("\x1B[33mWARNING\x1B[0m");
                break;
            case ERROR:
                fmt::print("\x1B[31m ERROR \x1B[0m");
                break;
            }
            fmt::print(" : {:<{}} ({:.3f}ms ± {:.3f})", test.testName, maxTestNameLength,
                       test.duration, test.std_derivation);

            if (!test.properties.empty()) {
                fmt::print(" [");
                bool first = true;
                for (const auto& prop : test.properties) {
                  if (!first) {
                    fmt::print(", ");
                  }
                  fmt::print("{} = {}", prop.name, prop.value);

                  first = false;
                }
                fmt::println("]");
            } else {
              fmt::println("");
            }
        }
    }

    g_testSuitResults.clear();
};
