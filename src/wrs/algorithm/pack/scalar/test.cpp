#include "./test.hpp"
#include "./test/test_cases.hpp"
#include "./test/test_setup.hpp"
#include "./test/test_types.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/pack/scalar/ScalarPack.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_alias_table.hpp"
#include "src/wrs/test/test.hpp"
#include "src/wrs/types/alias_table.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_structs.hpp>

using namespace wrs::test::scalar_pack;
using namespace wrs::test;

vk::DeviceSize wrs::test::scalar_pack::sizeOfWeight(const wrs::test::scalar_pack::WeightType ty) {
    switch (ty) {
    case WEIGHT_TYPE_FLOAT:
        return sizeof(float);
    }
    throw std::runtime_error("NOT IMPLEMENTED");
}

static void uploadPartitionIndicies(vk::CommandBuffer cmd,
                                    std::span<const wrs::glsl::uint> heavyIndicies,
                                    std::span<const wrs::glsl::uint> reverseLightIndicies,
                                    Buffers& buffers,
                                    Buffers& stage) {
    wrs::glsl::uint* indices = stage.heavyLightIndicies->get_memory()->map_as<wrs::glsl::uint>();
    std::memcpy(indices, heavyIndicies.data(), sizeof(wrs::glsl::uint) * heavyIndicies.size());
    std::memcpy(indices + heavyIndicies.size(), reverseLightIndicies.data(),
                sizeof(wrs::glsl::uint) * reverseLightIndicies.size());
    stage.heavyLightIndicies->get_memory()->unmap();
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer, {},
                        {},
                        stage.heavyLightIndicies->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                                 vk::AccessFlagBits::eTransferRead),
                        {});
    const std::size_t N = heavyIndicies.size() + reverseLightIndicies.size();
    vk::BufferCopy copy{0, 0, N * sizeof(wrs::glsl::uint)};
    cmd.copyBuffer(*stage.heavyLightIndicies, *buffers.heavyLightIndicies, 1, &copy);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        buffers.heavyLightIndicies->buffer_barrier(
                            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                        {});
}

template <wrs::arithmetic weight_t>
static void uploadSplits(vk::CommandBuffer cmd,
                         std::span<const wrs::Split<weight_t, wrs::glsl::uint>> splits,
                         Buffers& buffers,
                         Buffers& stage) {
    wrs::Split<weight_t, wrs::glsl::uint>* mapped =
        stage.splits->get_memory()->map_as<wrs::Split<weight_t, wrs::glsl::uint>>();
    std::memcpy(mapped, splits.data(),
                sizeof(wrs::Split<weight_t, wrs::glsl::uint>) * splits.size());
    stage.splits->get_memory()->unmap();
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer, {},
                        {},
                        stage.splits->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                     vk::AccessFlagBits::eTransferRead),
                        {});
    vk::BufferCopy copy{0, 0, sizeof(wrs::Split<weight_t, wrs::glsl::uint>) * splits.size()};
    cmd.copyBuffer(*stage.splits, *buffers.splits, 1, &copy);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        buffers.splits->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                       vk::AccessFlagBits::eShaderRead),
                        {});
}

template <wrs::arithmetic weight_t>
static void
downloadAliasTableToStage(vk::CommandBuffer cmd, std::size_t N, Buffers& buffers, Buffers& stage) {
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer, {}, {},
                        buffers.aliasTable->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                           vk::AccessFlagBits::eTransferRead),
                        {});

    vk::BufferCopy copy{0, 0, sizeof(wrs::AliasTableEntry<weight_t, wrs::glsl::uint>) * N};
    cmd.copyBuffer(*buffers.aliasTable, *stage.aliasTable, 1, &copy);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
                        {},
                        stage.aliasTable->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                         vk::AccessFlagBits::eHostRead),
                        {});
}

template <wrs::arithmetic weight_t>
static wrs::pmr::AliasTable<weight_t, wrs::glsl::uint>
downloadAliasTableFromStage(std::size_t N, Buffers& stage, std::pmr::memory_resource* resource) {
    const wrs::AliasTableEntry<weight_t, wrs::glsl::uint>* mapped =
        stage.aliasTable->get_memory()->map_as<wrs::AliasTableEntry<weight_t, wrs::glsl::uint>>();
    wrs::pmr::AliasTable<weight_t, wrs::glsl::uint> aliasTable{N, resource};
    std::memcpy(aliasTable.data(), mapped,
                sizeof(wrs::AliasTableEntry<weight_t, wrs::glsl::uint>) * N);
    stage.aliasTable->get_memory()->unmap();
    return aliasTable;
}

template <wrs::arithmetic weight_t>
static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName =
        fmt::format("{{weightCount={},dist={}splitCount={}}}", testCase.weightCount,
                    wrs::distribution_to_pretty_string(testCase.distribution), testCase.splitCount);
    SPDLOG_INFO("Running test case:{}", testName);

    SPDLOG_DEBUG("Creating ScalarPack instance");
    wrs::ScalarPack<weight_t> kernel{context.context};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        context.queue->wait_idle();
        if (testCase.iterations > 1) {
            if (testCase.weightCount > 1e6) {
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
        std::pmr::vector<weight_t> weights{resource};
        {
            weights = wrs::pmr::generate_weights<weight_t>(testCase.distribution, N, resource);
        }

        // 1.1 Compute reference input
        weight_t totalWeight;
        weight_t averageWeight;
        wrs::Partition<wrs::glsl::uint, std::pmr::vector<wrs::glsl::uint>> heavyLightIndicies;
        std::pmr::vector<weight_t> heavyPrefix{resource};
        std::pmr::vector<weight_t> lightPrefix{resource};
        std::pmr::vector<wrs::Split<weight_t, wrs::glsl::uint>> splits{resource};
        {
            totalWeight = wrs::reference::kahan_reduction<weight_t>(weights);
            averageWeight = totalWeight / static_cast<weight_t>(N);
            heavyLightIndicies = wrs::reference::pmr::stable_partition_indicies<weight_t, uint32_t>(
                weights, averageWeight, resource);

            const auto& deref = [&](const uint32_t i) -> weight_t { return weights[i]; };
            heavyPrefix = wrs::reference::pmr::prefix_sum<weight_t>(
                heavyLightIndicies.heavy() | std::views::transform(deref), resource);
            lightPrefix = wrs::reference::pmr::prefix_sum<weight_t>(
                heavyLightIndicies.light() | std::views::transform(deref), resource);

            splits = wrs::reference::pmr::splitK<weight_t, wrs::glsl::uint>(
                heavyPrefix, lightPrefix, averageWeight, N, K, resource);
        }
        const auto heavyIndices = heavyLightIndicies.heavy();
        std::pmr::vector<wrs::glsl::uint> reverseLightIndices{
            heavyLightIndicies.light().begin(), heavyLightIndicies.light().end(), resource};

        // 2. Begin recoding
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);

        // 3.0 Upload partition indices
        {
            uploadPartitionIndicies(cmd, heavyIndices, reverseLightIndices, buffers, stage);
        }
        // 3.1 Upload splits
        {
            uploadSplits<weight_t>(cmd, splits, buffers, stage);
        }
        // 4. Run test case
        {
            kernel.run(cmd, buffers);
        }

        // 5. Download results to stage
        {
            downloadAliasTableToStage<weight_t>(cmd, N, buffers, stage);
        }
        // 6. Submit to device
        {
            cmd.end();
            context.queue->submit_wait(cmd);
        }
        // 7. Download from stage
        wrs::pmr::AliasTable<weight_t, wrs::glsl::uint> aliasTable{resource};
        {
            aliasTable = downloadAliasTableFromStage<weight_t>(N, stage, resource);
        }
        // 8. Test pack invariants
        {
            const auto err =
                wrs::test::pmr::assert_is_alias_table<weight_t, weight_t, wrs::glsl::uint>(
                    weights, aliasTable, totalWeight, 0.01, resource);
            if (err) {
                SPDLOG_ERROR(err.message());
            }
        }
    }
    return failed;
}

void wrs::test::scalar_pack::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing scalar pack algorithm");
    TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{buffers.aliasTable->get_size() * 10};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        switch (testCase.weightType) {
        case WEIGHT_TYPE_FLOAT:
            bool failed = runTestCase<float>(testContext, testCase, buffers, stage, resource);
            if (failed) {
                failCount += 1;
            }
            break;
        }
        stackResource.reset();
    }
    if (failCount == 0) {
        SPDLOG_INFO("Scalar pack algorithm passed all tests");
    } else {
        SPDLOG_ERROR(fmt::format("Scalar pack algorithm failed {} out of {} tests", failCount,
                                 sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}
