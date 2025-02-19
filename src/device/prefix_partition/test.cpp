#include "./test.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/host/assert/is_prefix.hpp"
#include "src/device/prefix_partition/PrefixPartition.hpp"
#include "src/host/assert/is_stable_partition.hpp"
#include "src/host/assert/test.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/memory/FallbackResource.hpp"
#include "src/host/memory/SafeResource.hpp"
#include "src/host/memory/StackResource.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <memory_resource>
#include <spdlog/spdlog.h>

namespace device::test::prefix_partition {

using base = float;
using Algorithm = PrefixPartition<base>;
using Buffers = Algorithm::Buffers;
using Config = Algorithm::Config;

struct TestCase {
    Config config;
    host::glsl::uint N;
    host::Distribution distribution;
    base pivot;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .config = BlockWisePrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED),
        .N = 1024 * 2048,
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 1,
    },
    TestCase{ 
        .config = DecoupledPrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 32),
        .N = 1024 * 2048, 
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM, 
        .pivot = 0.5, 
        .iterations = 1, 
    }, 
};

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const base> elements,
                           base pivot) {
    {
        Buffers::ElementsView<base> stageView{stage.elements, elements.size()};
        Buffers::ElementsView<base> localView{buffers.elements, elements.size()};
        stageView.upload(elements);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        Buffers::PivotView<base> stageView{stage.pivot};
        Buffers::PivotView<base> localView{buffers.pivot};
        stageView.upload(pivot);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
}

static void downloadToStage(const merian::CommandBufferHandle& cmd,
                            Buffers& buffers,
                            Buffers& stage,
                            host::glsl::uint N) {
    {
        Buffers::PartitionIndicesView stageView{stage.partitionIndices, N};
        Buffers::PartitionIndicesView localView{buffers.partitionIndices, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::PartitionElementsView<base> stageView{stage.partitionElements, N};
        Buffers::PartitionElementsView<base> localView{buffers.partitionElements, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::PartitionPrefixView<base> stageView{stage.partitionPrefix, N};
        Buffers::PartitionPrefixView<base> localView{buffers.partitionPrefix, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::HeavyCountView stageView{stage.heavyCount};
        Buffers::HeavyCountView localView{buffers.heavyCount};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results2 {
    std::pmr::vector<host::glsl::uint> partitionIndices;
    std::pmr::vector<base> partitionElements;
    std::pmr::vector<base> partitionPrefix;
    host::glsl::uint heavyCount;
};
static Results2
downloadFromStage(Buffers& stage, host::glsl::uint N, std::pmr::memory_resource* resource) {
    Buffers::PartitionIndicesView partitionIndiciesView{stage.partitionIndices, N};
    Buffers::PartitionElementsView<base> partitionElementsView{stage.partitionElements, N};
    Buffers::PartitionPrefixView<base> partitionPrefixView{stage.partitionPrefix, N};
    Buffers::HeavyCountView heavyCountView{stage.heavyCount};

    auto partitionIndices =
        partitionIndiciesView.download<host::glsl::uint, host::pmr_alloc<host::glsl::uint>>(
            resource);
    auto partitionElements = partitionElementsView.download<base, host::pmr_alloc<base>>(resource);
    auto partitionPrefix = partitionPrefixView.download<base, host::pmr_alloc<base>>(resource);
    auto heavyCount = heavyCountView.download<host::glsl::uint>();

    return Results2{
        .partitionIndices = std::move(partitionIndices),
        .partitionElements = std::move(partitionElements),
        .partitionPrefix = std::move(partitionPrefix),
        .heavyCount = heavyCount,
    };
};

static bool runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{{},N={}}}", prefixPartitionConfigName(testCase.config), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Buffers buffers = Buffers::allocate<base>(context.alloc, merian::MemoryMappingType::NONE,
                                              testCase.config, testCase.N);

    Buffers stage = Buffers::allocate<base>(
        context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, testCase.config, testCase.N);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(context.profiler, testName);
        if (testCase.iterations > 1) {
            if (testCase.N > 1e6) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            }
        }

        // 1. Generate input
        context.profiler->start("Generate test input");
        const std::pmr::vector<base> elements =
            host::pmr::generate_weights<base>(testCase.distribution, testCase.N, resource);
        const base pivot = testCase.pivot;
        context.profiler->end();

        // 2. Begin recoding
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload test case indices
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload test case");
            SPDLOG_DEBUG("Uploading test case...");
            uploadTestCase(cmd, buffers, stage, elements, pivot);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N, context.profiler);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N);
        }

        // 6. Submit to device
        context.profiler->end();
        context.profiler->cmd_end(cmd);

        SPDLOG_DEBUG("Submitting to device...");
        cmd->end();
        context.queue->submit_wait(cmd);

        // 7. Download from stage
        context.profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results2 results = downloadFromStage(stage, testCase.N, resource);
        context.profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");

            if (testCase.N <= 1024) {
                fmt::println("PARTITION:");
                for (std::size_t i = 0; i < results.partitionIndices.size(); ++i) {
                    fmt::println("[{:>4}]: {:>12}     {:>12}     {:12}", i,
                                 results.partitionIndices[i], results.partitionElements[i],
                                 results.partitionPrefix[i]);
                }
                fmt::println("HEAVY-COUNT: {}", results.heavyCount);
            }

            if (testCase.N < (1 << 24)) {
                const std::span<float> heavy{results.partitionElements.begin(), results.heavyCount};

                const std::span<float> light{results.partitionElements.begin() + results.heavyCount,
                                             testCase.N - results.heavyCount};
                std::reverse(light.begin(), light.end());

                auto err = host::test::pmr::assert_is_stable_partition<float>(
                    heavy, light, elements, pivot, resource);
                if (err) {
                    SPDLOG_ERROR("{} constructs invalid partition: \n{}", testName, err.message());
                }


                const std::span<float> heavyPrefix{results.partitionPrefix.begin(), results.heavyCount};

                const std::span<float> lightPrefix{results.partitionPrefix.begin() + results.heavyCount,
                                             testCase.N - results.heavyCount};
                std::reverse(lightPrefix.begin(), lightPrefix.end());

                const auto err2 = host::test::pmr::assert_is_inclusive_prefix<base>(heavy, heavyPrefix, resource);
                
                if (err2) {
                  SPDLOG_ERROR("{} constructs invalid heavy prefix: \n {}", testName, err2.message());
                }

                const auto err3 = host::test::pmr::assert_is_inclusive_prefix<base>(light, lightPrefix, resource);

                if (err3) {
                  SPDLOG_ERROR("{} constructs invalid light prefix: \n {}", testName, err3.message());
                }


            }
            // TODO
        }
        context.profiler->collect(true, true);
        SPDLOG_DEBUG("EXIT test");
    }
    return failed;
}

void test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing prefix partition algorithm");

    const host::test::TestContext testContext = host::test::setupTestContext(context);

    host::memory::StackResource stackResource{4096 * 2048};
    host::memory::FallbackResource fallbackResource{&stackResource};
    host::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = std::pmr::get_default_resource();

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        runTestCase(testContext, testCase, resource);
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results (PrefixPartition): \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("All tests passed");
    } else {
        SPDLOG_ERROR(fmt::format("Failed {} out of {} tests", failCount,
                                 sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}

} // namespace device::test::prefix_partition
