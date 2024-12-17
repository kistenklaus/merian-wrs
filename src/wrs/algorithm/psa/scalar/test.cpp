#include "src/wrs/algorithm/psa/scalar/test.hpp"

#include <src/wrs/memory/FallbackResource.hpp>
#include <src/wrs/memory/SafeResource.hpp>
#include <src/wrs/memory/StackResource.hpp>
#include <src/wrs/reference/reduce.hpp>
#include <src/wrs/test/is_alias_table.hpp>
#include <src/wrs/test/test.hpp>
#include <src/wrs/types/alias_table.hpp>
#include <src/wrs/types/partition.hpp>

#include "ScalarPsa.hpp"
#include "./test_cases.hpp"

using Buffers = wrs::ScalarPsa::Buffers;
using weight_type = wrs::ScalarPsa::weight_t;

using namespace wrs::test::scalar_psa;

static void uploadWeights(vk::CommandBuffer cmd, const Buffers &buffers,
                          const Buffers &stage, const std::span<const weight_type> weights) {
    Buffers::WeightsView stageView{stage.weights, weights.size()};
    Buffers::WeightsView localView{buffers.weights, weights.size()};
    stageView.upload(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);
}

static void zeroDecoupledStates(const vk::CommandBuffer cmd,
                                const Buffers &buffers, const std::size_t weightCount) {
    constexpr std::size_t partitionSize = Buffers::DEFAULT_WORKGROUP_SIZE * Buffers::DEFAULT_ROWS;
    const std::size_t workgroupCount = (weightCount + partitionSize - 1) / partitionSize;
    Buffers::MeanDecoupledStateView meanStateView{buffers.meanDecoupledStates, workgroupCount};
    meanStateView.zero(cmd);
    meanStateView.expectComputeRead(cmd);
    Buffers::PartitionDecoupledStateView partitionStateView{buffers.partitionDecoupledState, workgroupCount};
    partitionStateView.zero(cmd);
    partitionStateView.expectComputeRead(cmd);
}

static void downloadAliasTableToStage(vk::CommandBuffer cmd, const Buffers &buffers,
                                      const Buffers &stage, const std::size_t weightCount) {
    Buffers::AliasTableView stageView{stage.aliasTable, weightCount};
    Buffers::AliasTableView localView{buffers.aliasTable, weightCount};
    localView.expectComputeWrite();
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

static wrs::pmr::AliasTable<weight_type, wrs::glsl::uint> downloadAliasTableFromStage(
    const Buffers &stage, const std::size_t weightCount, std::pmr::memory_resource *resource) {
    Buffers::AliasTableView stageView{stage.aliasTable, weightCount};

    using Entry = wrs::AliasTableEntry<weight_type, wrs::glsl::uint>;
    return stageView.download<Entry, wrs::pmr_alloc<Entry> >(resource);
}

static bool runTestCase(const wrs::test::TestContext &context,
                        const Buffers &buffers, const Buffers &stage, std::pmr::memory_resource *resource,
                        const TestCase &testCase) {
    std::string testName =
            fmt::format("{{weightCount={},dist={}splitCount={}}}", testCase.weightCount,
                        wrs::distribution_to_pretty_string(testCase.distribution), testCase.splitCount);
    SPDLOG_INFO("Running test case:{}", testName);

    SPDLOG_DEBUG("Creating ScalarPsa instance");
    wrs::ScalarPsa psa{context.context};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(context.profiler, testName);
        context.queue->wait_idle();
        if (testCase.iterations > 1) {
            if (testCase.weightCount > static_cast<std::size_t>(1e6)) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            }
        }
        const std::size_t N = testCase.weightCount;
        const std::size_t K = testCase.splitCount;

        // 1. Generate weights
        std::pmr::vector<weight_type> weights{resource}; {
            MERIAN_PROFILE_SCOPE(context.profiler, "Generate weights");
            SPDLOG_DEBUG("Generating weights...");
            weights = wrs::pmr::generate_weights<weight_type>(testCase.distribution, N, resource);
        }

        // 1.1 Compute reference input
        {
        }

        // 2. Begin recoding
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);

#ifdef MERIAN_PROFILER_ENABLE
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);
#endif

        // 3.0 Upload weights
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload weights indices");
            SPDLOG_DEBUG("Uploading weights...");
            uploadWeights(cmd, buffers, stage, weights);
            zeroDecoupledStates(cmd, buffers, N);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            psa.run(cmd, buffers, N, K, context.profiler);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadAliasTableToStage(cmd, buffers, stage, N);
        }
        // 6. Submit to device
        {
#ifdef MERIAN_PROFILER_ENABLE
            context.profiler->end();
            context.profiler->cmd_end(cmd);
            SPDLOG_DEBUG("Submitting to device...");
#endif
            cmd.end();
            context.queue->submit_wait(cmd);
        }
        // 7. Download from stage
        wrs::pmr::AliasTable<weight_type, wrs::glsl::uint> aliasTable{resource}; {
            SPDLOG_DEBUG("Downloading results from stage...");
            aliasTable = downloadAliasTableFromStage(stage, N, resource);
        }

        // 8. Test pack invariants
        {
            SPDLOG_DEBUG("Computing reference totalWeight with the kahan sum algorithm");
            const auto totalWeight = wrs::reference::kahan_reduction<weight_type>(weights);
            SPDLOG_DEBUG("Testing results");
            const auto err =
                    wrs::test::pmr::assert_is_alias_table<weight_type, weight_type, wrs::glsl::uint>(
                        weights, aliasTable, totalWeight, 0.01, resource);
            if (err) {
                SPDLOG_ERROR(err.message());
            }
        }

        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::scalar_psa::test(const merian::ContextHandle &context) {
    SPDLOG_INFO("Testing decoupled prefix partition algorithm");

    TestContext c = setupTestContext(context);

    std::size_t maxWeightCount = 0;
    std::size_t maxSplitCount = 0;
    for (const auto &testCase: TEST_CASES) {
        maxWeightCount = std::max(maxWeightCount, testCase.weightCount);
        maxSplitCount = std::max(maxSplitCount, testCase.splitCount);
    }

    auto buffers = Buffers::allocate(c.alloc, maxWeightCount,
                                     maxSplitCount, merian::MemoryMappingType::NONE);
    auto stage = Buffers::allocate(c.alloc, maxWeightCount,
                                   maxSplitCount, merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    memory::StackResource stackResource{buffers.weights->get_size() * 10};
    memory::FallbackResource fallbackResource{&stackResource};
    memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource *resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto &testCase: TEST_CASES) {
        if (runTestCase(c, buffers, stage, resource, testCase)) {
            failCount += 1;
        }
        stackResource.reset();
    }
    c.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
        merian::Profiler::get_report_str(c.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("decoupled prefix partition algorithm passed all tests");
    } else {
        SPDLOG_ERROR(fmt::format("decoupled prefix partition algorithm failed {} out of {} tests",
            failCount, sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}
