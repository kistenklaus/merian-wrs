#include "./test.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/wrs/alias/psa/PSA.hpp"
#include "src/device/wrs/alias/psa/layout/alias_table.hpp"
#include "src/device/wrs/alias/psa_plus/greedy/Greedy.hpp"
#include "src/device/wrs/alias/psa_plus/layout/heavy_light_count.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/reference/inverse_alias_table.hpp"
#include "src/host/test/context.hpp"
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <spdlog/spdlog.h>

#include "src/host/reference/reduce.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device::test::psa_greedy {

using Algorithm = PSAGreedy;
using Buffers = Algorithm::Buffers;
using Config = Algorithm::Config;
using weight_type = PSA::weight_type;

struct TestCase {
    Config config;
    host::glsl::uint N;
    host::Distribution distribution;
    uint32_t iterations;
};

static const TestCase TEST_CASES[] = {
    TestCase{
        .config = PSAGreedyConfig(4, 512),
        .N = static_cast<uint32_t>(1024 * 2048),
        .distribution = host::Distribution::PSEUDO_RANDOM_UNIFORM,
        .iterations = 1,
    },
};

static void uploadTestCase(const merian::CommandBufferHandle& cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> weights) {
    {
        Buffers::WeightsView stageView{stage.weights, weights.size()};
        Buffers::WeightsView localView{buffers.weights, weights.size()};
        stageView.upload(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        const auto totalWeight = host::reference::reduce(weights);
        float mean = totalWeight / weights.size();

        fmt::println("MEAN = {}", mean);
        Buffers::MeanView stageView{stage.mean};
        Buffers::MeanView localView{buffers.mean};
        stageView.upload(mean);
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
        Buffers::HeavyLightCountView stageView{stage.heavyLightCount};
        Buffers::HeavyLightCountView localView{buffers.heavyLightCount};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::WeightsView stageView{stage.weights, N};
        Buffers::WeightsView localView{buffers.weights, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::PartitionPrefixView stageView{stage.partitionPrefix, N};
        Buffers::PartitionPrefixView localView{buffers.partitionPrefix, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::AliasTableView stageView{stage.aliasTable, N};
        Buffers::AliasTableView localView{buffers.aliasTable, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::DebugView stageView{stage.debug, N};
        Buffers::DebugView localView{buffers.debug, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

struct Results {
    std::pmr::vector<host::glsl::uint> partitionIndices;
    std::pmr::vector<float> partitionPrefix;
    std::pmr::vector<float> weights;
    std::pmr::vector<host::glsl::uint> debug;
    std::pmr::vector<host::AliasTableEntry<weight_type, host::glsl::uint>> aliasTable;
    host::glsl::uint heavyCount;
    host::glsl::uint lightCount;
};
static Results
downloadFromStage(Buffers& stage, host::glsl::uint N, std::pmr::memory_resource* resource) {
    Buffers::PartitionIndicesView indicesView{stage.partitionIndices, N};
    auto partitionIndices =
        indicesView.download<host::glsl::uint, host::pmr_alloc<host::glsl::uint>>(resource);

    Buffers::PartitionPrefixView prefixView{stage.partitionPrefix, N};
    auto prefixPartition = prefixView.download<float, host::pmr_alloc<float>>(resource);

    Buffers::WeightsView weightsView{stage.weights, N};
    auto weights = weightsView.download<float, host::pmr_alloc<float>>(resource);

    auto aliasTable = device::details::downloadAliasTableFromBuffer(stage.aliasTable, N, resource);

    auto [heavyCount, lightCount] =
        device::details::downloadHeavyLightCountFromStage(stage.heavyLightCount);

    Buffers::DebugView stageView{stage.debug, N};
    auto debug = stageView.download<host::glsl::uint, host::pmr_alloc<host::glsl::uint>>(resource);

    return Results{
        .partitionIndices = std::move(partitionIndices),
        .partitionPrefix = std::move(prefixPartition),
        .weights = std::move(weights),
        .debug = std::move(debug),
        .aliasTable = std::move(aliasTable),
        .heavyCount = heavyCount,
        .lightCount = lightCount,
    };
};

static bool runTestCase(const host::test::TestContext& context,
    const merian::ProfilerHandle& profiler,
                        const TestCase& testCase,
                        std::pmr::memory_resource* resource) {
    Buffers buffers = Buffers::allocate(context.alloc, testCase.N, testCase.config.blockSize(),
                                        merian::MemoryMappingType::NONE);
    Buffers stage = Buffers::allocate(context.alloc, testCase.N, testCase.config.blockSize(),
                                      merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    std::string testName =
        fmt::format("Greedy:{{{},N={}}}", testCase.config.blockSize(), testCase.N);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, context.shaderCompiler, testCase.config};

    bool failed = false;
    double averageJSDivergence = 0;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(profiler, testName);
        context.queue->wait_idle();
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
        auto weights =
            host::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
        profiler->end();

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(context.cmdPool);
        cmd->begin();

        // 2. Begin recoding
        /* std::string recordingLabel = fmt::format("Recording : {}", testName); */
        /* context.profiler->start(recordingLabel); */
        /* context.profiler->cmd_start(cmd, recordingLabel); */

        // 3. Upload test case indices
        {
            {
                /* MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload test case"); */
                SPDLOG_DEBUG("Uploading test case...");
                uploadTestCase(cmd, buffers, stage, weights);
            }
        }

        {
            {
                /* MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "PSA"); */

                profiler->start("PSA-Greedy");
                profiler->cmd_start(cmd, "PSA-Greedy");

                kernel.run(cmd, buffers, testCase.N, profiler);

                profiler->end();
                profiler->cmd_end(cmd);
            }
        }

        // Submit to device

        // Download results to stage
        {

            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eTransfer,
                         buffers.weights->buffer_barrier(vk::AccessFlagBits::eShaderRead,
                                                         vk::AccessFlagBits::eTransferWrite));

            /* MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage"); */
            /* SPDLOG_DEBUG("Downloading results to stage..."); */
            downloadToStage(cmd, buffers, stage, testCase.N);
        }
        cmd->end();
        context.queue->submit_wait(cmd);

        // Download from stage
        profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results results = downloadFromStage(stage, testCase.N, resource);

        profiler->end();

        // Test results
        {
            MERIAN_PROFILE_SCOPE(profiler, "Testing results");
            float mean = host::reference::reduce<float>(weights) / weights.size();

            auto normWeights = host::reference::normalize_weights<float>(weights);
            auto normWeightsOut = host::reference::normalize_weights<float>(results.weights, mean);
            auto sampledWeights =
                host::reference::alias_table_to_normalized_weights<float, host::glsl::uint>(
                    results.aliasTable);
            if (testCase.N < 1024 * 2048) {
                for (std::size_t i = 0; i < results.debug.size(); ++i) {
                    fmt::println("[{}]: {}", i, results.debug[i]);
                }
            }

            std::size_t lx = 0;
            std::size_t hx = 0;

            for (std::size_t i = 0; i < results.debug.size(); i += 4) {
                hx += results.debug[i];
                lx += results.debug[i + 1];
            }

            std::size_t packed = 0;
            if (testCase.N < 1024 * 2048) {
                fmt::println("ALIAS-TABLE:");
                for (std::size_t i = 0; i < results.aliasTable.size(); ++i) {
                    fmt::println("[{:>3}]: ({:.4f}) ~ ({:.4f}) -> ({:.4f},{:>3}) -> ({:.4f})", i,
                                 normWeights[i], normWeightsOut[i], results.aliasTable[i].p,
                                 results.aliasTable[i].a, sampledWeights[i]);
                    if (results.aliasTable[i].a != 0 && results.aliasTable[i].p != 0.0f) {
                        packed += 1;
                    }
                }
            }

            if (testCase.N < 1024 * 2048) {
                fmt::println("PARTITION:");
                for (std::size_t i = 0; i < results.partitionIndices.size(); ++i) {
                    fmt::println("[{:>3}]: {:>3}    {:.4f}", i, results.partitionIndices[i],
                                 results.partitionPrefix[i] / mean);
                }
            }
            fmt::println("LIGHT-COUNT: {} ({})   HEAVY-COUNT: {} ({})  greedily-packed: {}   ({})",
                         results.lightCount, lx, results.heavyCount, hx,
                         (testCase.N - (results.heavyCount + results.lightCount)),
                         testCase.N - packed);
        }
    }
    profiler->collect(true, true);

    averageJSDivergence /= testCase.iterations;
    SPDLOG_INFO("JS-Divergence: {}", averageJSDivergence);

    if (averageJSDivergence > 0.3) {
        SPDLOG_ERROR("{} - is sampling something completely different", testName);
    } else if (averageJSDivergence > 0.15) {
        SPDLOG_ERROR("{} displays a significant bias", testName);
    } else if (averageJSDivergence > 0.05) {
        SPDLOG_WARN("{} displays a moderate bias", testName);
    } else {
        SPDLOG_INFO("{} is does not show any significant bias", testName);
    }
    return failed;
}

void test(const host::test::TestContext& context) {
    SPDLOG_INFO("Testing PSA-Greedy algorithm");

    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context.context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context.context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    std::pmr::memory_resource* resource = context.memory_resource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        runTestCase(context, profiler, testCase, resource);
    }

    profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("All tests passed");
    } else {
        SPDLOG_ERROR(fmt::format("Failed {} out of {} tests", failCount,
                                 sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}

} // namespace device::test::psa_greedy
