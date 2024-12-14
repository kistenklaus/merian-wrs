#include "./test.hpp"
#include "./test/test_cases.hpp"
#include "./test/test_setup.hpp"
#include "./test/test_types.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/algorithm/pack/scalar/ScalarPack.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/generic_types.hpp"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/test.hpp"
#include <cstring>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>

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
                         std::span<const wrs::split_t<weight_t, wrs::glsl::uint>> splits,
                         Buffers& buffers,
                         Buffers& stage) {
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
        weight_t averageWeight;
        std::pmr::vector<wrs::glsl::uint> partitionIndexStorage{resource};
        std::span<wrs::glsl::uint> heavyPartitionIndicies;
        std::span<wrs::glsl::uint> lightPartitionIndicies;
        std::pmr::vector<weight_t> heavyPrefix{resource};
        std::pmr::vector<weight_t> lightPrefix{resource};
        std::pmr::vector<wrs::split_t<weight_t, wrs::glsl::uint>> splits{resource};
        {
            weight_t totalWeight = wrs::reference::kahan_reduction<weight_t>(weights);
            averageWeight = totalWeight / static_cast<weight_t>(N);
            auto [heavyIndices, lightIndicies, indexStorage] =
                wrs::reference::pmr::stable_partition_indicies<weight_t, uint32_t>(
                    weights, averageWeight, resource);
            partitionIndexStorage.swap(indexStorage);
            heavyPartitionIndicies = heavyIndices;
            lightPartitionIndicies = lightIndicies;

            const auto& deref = [&](const uint32_t i) -> weight_t { return weights[i]; };
            heavyPrefix = wrs::reference::pmr::prefix_sum<weight_t>(
                heavyPartitionIndicies | std::views::transform(deref), resource);
            lightPrefix = wrs::reference::pmr::prefix_sum<weight_t>(
                lightPartitionIndicies | std::views::transform(deref), resource);

            splits = wrs::reference::pmr::splitK<weight_t, wrs::glsl::uint>(
                heavyPrefix, lightPrefix, averageWeight, N, K, resource);
        }

        // 2. Begin recoding
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);

        // 3.0 Upload partition indices
        {
        }
        // 3.1 Upload partition prefixes
        {
        }
        // 4. Run test case
        {
            kernel.run(cmd, buffers);
        }

        // 5. Download results to stage
        {
        }
        // 6. Submit to device
        {
        }
        // 7. Download from stage
        {
        }
        // 8. Test pack invariants
        {
        }
    }
    return failed;
}

void wrs::test::scalar_pack::test(const merian::ContextHandle& context) {
    TestContext testContext = setupTestContext(context);

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
