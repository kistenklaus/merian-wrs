#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_partition/PrefixPartition.hpp"
#include "src/host/assert/is_prefix.hpp"
#include "src/host/assert/is_stable_partition.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/test/context.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory_resource>
#include <spdlog/spdlog.h>

#ifndef MERIAN_PROFILER_ENABLE
#define MERIAN_PROFILER_ENABLE
#endif

namespace device::prefix_partition {

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

static const TestCase TEST_CASES[] = {
    //

    TestCase{
        .config = BlockWisePrefixPartitionConfig(512, 2, BlockScanVariant::RANKED_STRIDED),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 2,
    },
    TestCase{
        .config = BlockWisePrefixPartitionConfig(512, 4, BlockScanVariant::RANKED_STRIDED),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 2,
    },
    TestCase{
        .config = BlockWisePrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 1),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 2,
    },
    TestCase{
        .config = BlockWisePrefixPartitionConfig(512, 8, BlockScanVariant::RANKED_STRIDED, 2),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 1,
    },

    TestCase{
        .config = DecoupledPrefixPartitionConfig(),
        .N = static_cast<uint32_t>(1e4),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPrefixPartitionConfig(),
        .N = static_cast<uint32_t>(1e5),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPrefixPartitionConfig(),
        .N = static_cast<uint32_t>(1e6),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 2,
    },
    TestCase{
        .config = DecoupledPrefixPartitionConfig(),
        .N = static_cast<uint32_t>(1e7),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 1,
    },
    TestCase{
        .config = DecoupledPrefixPartitionConfig(),
        .N = static_cast<uint32_t>(1e8),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .pivot = 0.5,
        .iterations = 1,
    },

    //
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
}

static void runTestCase(const host::test::TestContext& context,
                        const TestCase& testCase,
                        std::pmr::memory_resource* resource) {

    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(context.queue);

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context.context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context.context, 4096);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    std::string testName =
        fmt::format("{{{},N={}}}", prefixPartitionConfigName(testCase.config), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    auto alloc = context.createResourceAllocator();

    Buffers buffers = Buffers::allocate<base>(alloc, merian::MemoryMappingType::NONE,
                                              testCase.config, testCase.N);

    Buffers stage = Buffers::allocate<base>(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM,
                                            testCase.config, testCase.N);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config, true};

    std::string recordingLabel = fmt::format("Recording : {}", testName);

    if (kernel.maxElementCount() < testCase.N) {
        throw std::runtime_error(fmt::format("Input size is to large {} > {} = N",
                                             kernel.maxElementCount(), testCase.N));
    }

    host::test::TestResultType out = host::test::SUCCESS;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(profiler, testName);
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
        profiler->start("Generate test input");
        const std::pmr::vector<base> elements =
            host::pmr::generate_weights<base>(testCase.distribution, testCase.N, resource);
        const base pivot = testCase.pivot;
        profiler->end();

        // 2. Begin recoding
        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();
        profiler->start(recordingLabel);
        profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload test case indices
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Upload test case");
            SPDLOG_DEBUG("Uploading test case...");
            uploadTestCase(cmd, buffers, stage, elements, pivot);
        }

        // 4. Run test case
        {
            SPDLOG_DEBUG("Execute algorithm");
            profiler->start("Execute algorithm");
            profiler->cmd_start(cmd, "Execute algorithm",
                                vk::PipelineStageFlagBits::eComputeShader);
            kernel.run(cmd, buffers, testCase.N, profiler);
            profiler->end();
            profiler->cmd_end(cmd, vk::PipelineStageFlagBits::eComputeShader);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N);
        }

        // 6. Submit to device
        profiler->end();
        profiler->cmd_end(cmd);

        SPDLOG_DEBUG("Submitting to device...");
        cmd->end();
        context.queue->submit_wait(cmd);

        // 7. Download from stage
        profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results2 results = downloadFromStage(stage, testCase.N, resource);
        profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(profiler, "Testing results");
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

            const std::span<float> heavy{results.partitionElements.begin(), results.heavyCount};

            const std::span<float> light{results.partitionElements.begin() + results.heavyCount,
                                         testCase.N - results.heavyCount};
            std::reverse(light.begin(), light.end());

            if (testCase.N <= 1e7) {
                auto err = host::test::pmr::assert_is_stable_partition<float>(
                    heavy, light, elements, pivot, resource);
                if (err) {
                    SPDLOG_ERROR("{} constructs invalid partition: \n{}", testName, err.message());
                    out += host::test::ERROR;
                }
            } else {
                SPDLOG_INFO("Skipping full verification of partitions. No pretty error messages");
            }

            for (std::size_t i = 0; i < results.heavyCount; ++i) {
                if (heavy[i] != elements[results.partitionIndices[i]]) {
                    SPDLOG_ERROR("Partition indices are not correct");
                    out += host::test::ERROR;
                    break;
                }
            }

            for (std::size_t i = 0; i < (testCase.N - results.heavyCount); ++i) {
                if (light[i] != elements[results.partitionIndices[testCase.N - 1 - i]]) {
                    SPDLOG_ERROR("Partition indices are not correct");
                    out += host::test::ERROR;
                    break;
                }
            }

            const std::span<float> heavyPrefix{results.partitionPrefix.begin(), results.heavyCount};

            const std::span<float> lightPrefix{results.partitionPrefix.begin() + results.heavyCount,
                                               testCase.N - results.heavyCount};
            std::reverse(lightPrefix.begin(), lightPrefix.end());

            const auto err2 =
                host::test::pmr::assert_is_inclusive_prefix<base>(heavy, heavyPrefix, resource);

            if (err2) {
                SPDLOG_WARN("{} constructs invalid heavy prefix: \n {}", testName, err2.message());
                out += host::test::WARNING;
            }

            const auto err3 =
                host::test::pmr::assert_is_inclusive_prefix<base>(light, lightPrefix, resource);

            if (err3) {
                SPDLOG_WARN("{} constructs invalid light prefix: \n {}", testName, err3.message());
                out += host::test::WARNING;
            }
        }
        profiler->collect(true, true);
    }

    profiler->collect(true, false);

    const auto report = profiler->get_report().gpu_report;
    auto recordingEntry =
        std::ranges::find_if(report, [&](const merian::Profiler::ReportEntry& entry) {
            return entry.name == recordingLabel;
        });
    if (recordingEntry == report.end()) {
        fmt::println("ENTRIES:");
        for (const auto& x : report) {
            fmt::println("{}", x.name);
        }
        throw std::runtime_error("Impossible state 1");
    }
    auto entry = std::ranges::find_if(recordingEntry->children,
                                      [&](const merian::Profiler::ReportEntry& entry) {
                                          return entry.name == "Execute algorithm";
                                      });
    if (entry == recordingEntry->children.end()) {
        throw std::runtime_error("Impossible state 2");
    }

    context.pushResult(
        "ScanPartition", prefixPartitionConfigClass(testCase.config), out, entry->duration,
        entry->std_deviation,
        {host::test::TestProperty{.name = "N",
                                  .value = std::format("{:.0e}", static_cast<float>(testCase.N))}});

    SPDLOG_INFO(fmt::format("Profiler results (PrefixPartition): \n{}",
                            merian::Profiler::get_report_str(profiler->get_report())));

}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing prefix partition algorithm");

    std::pmr::memory_resource* resource = context.memory_resource;

    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, testCase, resource);
    }
}

} // namespace device::prefix_partition
