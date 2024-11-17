#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/alias_table/baseline/kernels/DecoupledPartitionAndPrefixSumKernel.hpp"
#include "src/wrs/cpu/stable.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_enums.hpp>
namespace wrs::baseline {

class PartitionAndPrefixSum {
    using weight_t = float;

  public:
    PartitionAndPrefixSum(const merian::ContextHandle& context,
                          uint32_t maxElementCount,
                          uint32_t workgroupSize = 512,
                          uint32_t rows = 5)
        : m_partitionAndPrefixSumKernel(context, workgroupSize, rows) {

        auto resources = context->get_extension<merian::ExtensionResources>();
        assert(resources != nullptr);
        auto alloc = resources->resource_allocator();
        uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
        uint32_t maxWorkgroupCount = (maxElementCount + partitionSize - 1) / partitionSize;
        m_partitionDescriptorBuffer = alloc->createBuffer(
            m_partitionAndPrefixSumKernel.partitionDescriptorBufferSize(maxWorkgroupCount),
            vk::BufferUsageFlagBits::eStorageBuffer, merian::MemoryMappingType::NONE);
    }

    void reset(vk::CommandBuffer cmd, std::optional<uint32_t> elementCount) {
        vk::DeviceSize size;
        if (elementCount.has_value()) {
            uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
            uint32_t workgroupCount = (elementCount.value() + partitionSize - 1) / partitionSize;
            size = m_partitionAndPrefixSumKernel.partitionDescriptorBufferSize(workgroupCount);
        } else {
            size = VK_WHOLE_SIZE;
        }
        cmd.fillBuffer(*m_partitionDescriptorBuffer, 0, size, 0);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {},
            m_partitionDescriptorBuffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                        vk::AccessFlagBits::eShaderRead),
            {});
    }

    void run(vk::CommandBuffer cmd,
             merian::BufferHandle in_elements,
             merian::BufferHandle in_avgPrefixSum,
             merian::BufferHandle out_partitions,
             merian::BufferHandle out_partitionPrefix,
             std::optional<uint32_t> elementCount = std::nullopt,
             std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        uint32_t count = elementCount.value_or(in_elements->get_size() / sizeof(float));
        uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
        uint32_t workgroupCount = (count + partitionSize - 1) / partitionSize;

#ifdef MERIAN_PROFILER_ENABLE
        std::optional<merian::ProfileScopeGPU> merian_profile_scope = std::nullopt;
        if (profiler.has_value()) {
            merian_profile_scope.emplace(
                profiler.value(), cmd,
                fmt::format(
                    "decoupled partition and prefix sum [workgroupCount = {}, optimal={:.3}ms]",
                    workgroupCount, getOptimalTime(count).count()));
        }
#endif

        m_partitionAndPrefixSumKernel.dispatch(cmd, workgroupCount, in_elements, count,
                                               in_avgPrefixSum, out_partitions, out_partitionPrefix,
                                               m_partitionDescriptorBuffer);
    }

    std::chrono::duration<float, std::milli> getOptimalTime(uint32_t elementCount) {
        constexpr uint32_t expectedLookBacks = 4;
        uint32_t partitionSize = m_partitionAndPrefixSumKernel.partitionSize();
        uint32_t workgroupCount = (elementCount + partitionSize - 1) / partitionSize;
        float requiredTransfer =
            elementCount * (sizeof(weight_t) + sizeof(weight_t) +
                            sizeof(weight_t)) // read weights, write partition and prefix
            + sizeof(weight_t) +              // read pivot
            workgroupCount *
                (sizeof(unsigned int) + 2 * sizeof(weight_t) + 2 * sizeof(unsigned int)) *
                expectedLookBacks;               // lookback accesses
        constexpr float memoryBandwidth = 504e9; // RTX 4070
        return std::chrono::duration<float, std::milli>((requiredTransfer / memoryBandwidth) * 1e3);
    }

    static void testAndBench(const merian::ContextHandle& context) {
        SPDLOG_INFO("Setup tests and benchmark");
        struct PartitionAndPrefixSumConfig {
            uint32_t workgroupSize;
            uint32_t rows;
        };

        constexpr std::array<PartitionAndPrefixSumConfig, 1> testConfigurations = {
            /* PartitionAndPrefixSumConfig{512, 1}, PartitionAndPrefixSumConfig{512, 2}, */
            PartitionAndPrefixSumConfig{32, 1}, 
            /* PartitionAndPrefixSumConfig{512, 5}, */
            /* PartitionAndPrefixSumConfig{512, 8}, PartitionAndPrefixSumConfig{512, 16}, */
            /* PartitionAndPrefixSumConfig{512, 32}, */
            /* PartitionAndPrefixSumConfig{512, 64}, */
            /* PartitionAndPrefixSumConfig{512, 128}, */
            /* PartitionAndPrefixSumConfig{512, 256}, */
        };
        constexpr std::array<WeightGenInfo, 14> weightGens = {
            WeightGenInfo{RANDOM_UNIFORM, 8},
            WeightGenInfo{RANDOM_UNIFORM, 12},
            WeightGenInfo{RANDOM_UNIFORM, 16},
            WeightGenInfo{RANDOM_UNIFORM, 24},
            WeightGenInfo{RANDOM_UNIFORM, 32},
            WeightGenInfo{RANDOM_UNIFORM, 128},
            WeightGenInfo{RANDOM_UNIFORM, 256},
            WeightGenInfo{RANDOM_UNIFORM, 512},
            WeightGenInfo{RANDOM_UNIFORM, 1024},
            WeightGenInfo{RANDOM_UNIFORM, 2048},
            WeightGenInfo{SEEDED_RANDOM_UNIFORM, 4096},
            WeightGenInfo{SEEDED_RANDOM_UNIFORM, 4096 * 2},
            WeightGenInfo{SEEDED_RANDOM_UNIFORM, 4096 * 8},
            WeightGenInfo{SEEDED_RANDOM_UNIFORM, 4096 * 64},
            /* WeightGenInfo{SEEDED_RANDOM_UNIFORM, 2048 * 1024}, */
            /* WeightGenInfo{SEEDED_RANDOM_UNIFORM, 1000000}, */
            /* WeightGenInfo{SEEDED_RANDOM_UNIFORM, 2000000}, */
            /* WeightGenInfo{SEEDED_RANDOM_UNIFORM, 3000000}, */
            /* WeightGenInfo{SEEDED_RANDOM_UNIFORM, 4000000}, */
            /* WeightGenInfo{SEEDED_RANDOM_UNIFORM, 5000000}, */
        };

        // Setup queue & command pool
        auto resources = context->get_extension<merian::ExtensionResources>();
        assert(resources != nullptr);
        merian::ResourceAllocatorHandle alloc = resources->resource_allocator();

        merian::QueueHandle queue = context->get_queue_GCT();
        merian::CommandPool cmdPool(queue);

        SPDLOG_DEBUG("Setting up resources for testing");
        merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
        merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
            std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
        query_pool->reset();
        profiler->set_query_pool(query_pool);

        // Create resources (buffers)
        const uint32_t maxWeightCount =
            std::max_element(
                weightGens.begin(), weightGens.end(),
                [](const WeightGenInfo& a, const WeightGenInfo& b) { return a.count < b.count; })
                ->count;
        merian::BufferHandle weightBuffer = alloc->createBuffer(
            maxWeightCount * sizeof(weight_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            merian::MemoryMappingType::NONE);
        merian::BufferHandle weightBufferStage =
            alloc->createBuffer(weightBuffer->get_size(), vk::BufferUsageFlagBits::eTransferSrc,
                                merian::MemoryMappingType::HOST_ACCESS_RANDOM);

        merian::BufferHandle avgPrefixBuffer = alloc->createBuffer(
            sizeof(uint32_t) + maxWeightCount * sizeof(weight_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            merian::MemoryMappingType::NONE);
        merian::BufferHandle avgPrefixBufferStage =
            alloc->createBuffer(avgPrefixBuffer->get_size(), vk::BufferUsageFlagBits::eTransferSrc,
                                merian::MemoryMappingType::HOST_ACCESS_RANDOM);

        merian::BufferHandle partitionBuffer = alloc->createBuffer(
            sizeof(uint32_t) + maxWeightCount * sizeof(weight_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            merian::MemoryMappingType::NONE);
        merian::BufferHandle partitionBufferStage =
            alloc->createBuffer(partitionBuffer->get_size(), vk::BufferUsageFlagBits::eTransferDst,
                                merian::MemoryMappingType::HOST_ACCESS_RANDOM);

        merian::BufferHandle partitionPrefixBuffer = alloc->createBuffer(
            weightBuffer->get_size(),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            merian::MemoryMappingType::NONE);
        merian::BufferHandle partitionPrefixBufferStage = alloc->createBuffer(
            partitionPrefixBuffer->get_size(), vk::BufferUsageFlagBits::eTransferDst,
            merian::MemoryMappingType::HOST_ACCESS_RANDOM);

        struct TestResult {
            WeightGenInfo weightInfo;
            bool succ;
        };
        std::vector<std::tuple<PartitionAndPrefixSumConfig, std::vector<TestResult>>> results;
        for (const auto& algoConfig : testConfigurations) {
            MERIAN_PROFILE_SCOPE(
                profiler, fmt::format("Algorithm configuration [workgroupSize = {}, rows = {}]",
                                      algoConfig.workgroupSize, algoConfig.rows));
            std::vector<TestResult> configResults;
            SPDLOG_DEBUG("Construct algorithm instance");
            PartitionAndPrefixSum algo(context, maxWeightCount, algoConfig.workgroupSize,
                                       algoConfig.rows);
            bool success = true;
            for (const auto weightGen : weightGens) {
                vk::CommandBuffer cmd = cmdPool.create_and_begin();

                MERIAN_PROFILE_SCOPE(
                    profiler, fmt::format("Test configuration [distribution = {}, count = {}]",
                                          distribution_to_pretty_string(weightGen.distribution),
                                          algoConfig.rows));

                SPDLOG_INFO(fmt::format("Begin testcase [workgroupSize={}, "
                                        "rows={}, dist={}, count={}]",
                                        algoConfig.workgroupSize, algoConfig.rows,
                                        distribution_to_pretty_string(weightGen.distribution),
                                        weightGen.count));
                SPDLOG_DEBUG("Generate weights");
                std::vector<weight_t> weights = generate_weights(weightGen);

                SPDLOG_DEBUG("compute (percise) prefix sum and average");
                std::vector<double> d_weights(weights.size());
                for (size_t i = 0; i < weights.size(); ++i) {
                    d_weights[i] = weights[i];
                }

                std::vector<double> d_weightPrefix = wrs::cpu::stable::prefix_sum(d_weights);
                std::vector<weight_t> weightPrefix = wrs::cpu::stable::prefix_sum(weights);

                double d_average = d_weightPrefix.back() / static_cast<double>(weights.size());
                weight_t average = static_cast<weight_t>(d_average);
                /* sleep(2); */

                SPDLOG_DEBUG("Upload weights, average and prefix sum");
                {
                    // Upload weights
                    {
                        void* weightsMapped = weightBufferStage->get_memory()->map();
                        std::memcpy(weightsMapped, weights.data(),
                                    weights.size() * sizeof(weight_t));
                        weightBufferStage->get_memory()->unmap();
                        vk::BufferCopy copy{0, 0, weights.size() * sizeof(weight_t)};
                        cmd.pipelineBarrier(
                            vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                            {}, {},
                            weightBufferStage->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                              vk::AccessFlagBits::eTransferRead),
                            {});
                        cmd.copyBuffer(*weightBufferStage, *weightBuffer, 1, &copy);
                    }

                    // Upload average and prefix sum
                    {
                        uint8_t* avgPrefixMapped =
                            avgPrefixBufferStage->get_memory()->map_as<uint8_t>();
                        float* avgMapped = reinterpret_cast<weight_t*>(avgPrefixMapped);
                        float* prefixMapped =
                            reinterpret_cast<weight_t*>(avgPrefixMapped + sizeof(weight_t));
                        *avgMapped = average;
                        std::memcpy(prefixMapped, weightPrefix.data(),
                                    weightPrefix.size() * sizeof(weight_t));
                        avgPrefixBufferStage->get_memory()->unmap();
                        cmd.pipelineBarrier(
                            vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                            {}, {},
                            avgPrefixBufferStage->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                                 vk::AccessFlagBits::eTransferRead),
                            {});
                        vk::BufferCopy copy{
                            0, 0, sizeof(uint32_t) + weightPrefix.size() * sizeof(weight_t)};
                        cmd.copyBuffer(*avgPrefixBufferStage, *avgPrefixBuffer, 1, &copy);
                    }
                }

                // Pipeline barrier
                {
                    cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        {
                            weightBuffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                         vk::AccessFlagBits::eShaderRead),
                            avgPrefixBuffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                            vk::AccessFlagBits::eShaderRead),
                        },
                        {});
                }
                {
                    algo.reset(cmd, weights.size());
                }

                SPDLOG_DEBUG("Execute algorithm");
                {
                    MERIAN_PROFILE_SCOPE_GPU(
                        profiler, cmd,
                        fmt::format("Test configuration [distribution = {}, weightCount = {}, "
                                    "workgroupSize = {}, rows = {}]",
                                    distribution_to_pretty_string(weightGen.distribution),
                                    weightGen.count, algoConfig.workgroupSize, algoConfig.rows));
                    algo.run(cmd, weightBuffer, avgPrefixBuffer, partitionBuffer,
                             partitionPrefixBuffer, weights.size(), profiler);
                }
                // Pipeline barrier
                {
                    cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer, {}, {},
                        {
                            partitionBuffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eTransferRead),
                            partitionPrefixBuffer->buffer_barrier(
                                vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eTransferRead),
                        },
                        {});
                }
                SPDLOG_DEBUG("Downloading results");
                {
                    {
                        vk::BufferCopy copy{0, 0,
                                            sizeof(uint32_t) + weights.size() * sizeof(weight_t)};
                        cmd.copyBuffer(*partitionBuffer, *partitionBufferStage, 1, &copy);
                    }
                    {
                        vk::BufferCopy copy{0, 0, weights.size() * sizeof(weight_t)};
                        cmd.copyBuffer(*partitionPrefixBuffer, *partitionPrefixBufferStage, 1,
                                       &copy);
                    }
                }
                { // Pipeline barrier
                    cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
                        {},
                        {
                            partitionBufferStage->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                                 vk::AccessFlagBits::eHostRead),
                            partitionPrefixBufferStage->buffer_barrier(
                                vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead),
                        },
                        {});
                }
                cmd.end();
                queue->submit_wait(cmd);

                std::vector<weight_t> resultPartitionLight;
                std::vector<weight_t> resultPartitionHeavy;
                uint32_t heavyCount;
                uint32_t lightCount;

                { // Download from mapped memory
                    {
                        uint8_t* partitionAndCountMapped =
                            partitionBufferStage->get_memory()->map_as<uint8_t>();
                        heavyCount = *reinterpret_cast<uint32_t*>(partitionAndCountMapped);
                        uint8_t* partitionMapped = partitionAndCountMapped + sizeof(uint32_t);

                        if (heavyCount > weights.size()) {
                            SPDLOG_ERROR(
                                fmt::format("HeavyCount is greater than the "
                                            "weightCount. [heavyCount = {}, weightCount = {}]",
                                            heavyCount, weights.size()));
                            SPDLOG_ERROR("FATAL ERROR KILLING PROCESS");
                            exit(-1); // BAD IMPLEMENTATION
                        }
                        lightCount = weights.size() - heavyCount;
                        resultPartitionHeavy.resize(heavyCount);
                        resultPartitionLight.resize(lightCount);
                        std::memcpy(resultPartitionHeavy.data(), partitionMapped,
                                    heavyCount * sizeof(weight_t));
                        std::memcpy(resultPartitionLight.data(),
                                    partitionMapped + heavyCount * sizeof(weight_t),
                                    lightCount * sizeof(weight_t));
                        std::reverse(resultPartitionLight.begin(), resultPartitionLight.end());

                        partitionBufferStage->get_memory()->unmap();
                    }
                }

                std::vector<weight_t> resultPartitionPrefixHeavy(heavyCount);
                std::vector<weight_t> resultPartitionPrefixLight(lightCount);
                {
                    float* partitionPrefixMapped =
                        partitionPrefixBufferStage->get_memory()->map_as<float>();
                    if (heavyCount > 0) {
                        std::memcpy(resultPartitionPrefixHeavy.data(), partitionPrefixMapped,
                                    heavyCount * sizeof(weight_t));
                    }
                    std::memcpy(resultPartitionPrefixLight.data(),
                                partitionPrefixMapped + heavyCount, lightCount * sizeof(weight_t));
                    partitionPrefixBufferStage->get_memory()->unmap();
                    std::reverse(resultPartitionPrefixLight.begin(),
                                 resultPartitionPrefixLight.end());
                }

                /* SPDLOG_LOGGER_CALL(logger, level, ...) */
                SPDLOG_DEBUG("Checking result invariants");
                { // Test partition invariants
                    // Invariant: lightCount + heavyCount == weightCount
                    if (heavyCount + lightCount != weights.size()) {
                        success = false;
                        SPDLOG_ERROR(
                            fmt::format("Broken Invariant : heavyCount + "
                                        "lightCount == weightCount."
                                        "[heavyCount = {}, lightCount = {}, weightCount = {}]",
                                        heavyCount, lightCount, weights.size()));
                    }
                    // Invariant: ForAll: light <= average
                    {
                        uint32_t brokenCount = 0;
                        uint32_t maxBrokenCountLog = 4;
                        for (size_t i = 0; i < lightCount; ++i) {
                            if (resultPartitionLight[i] > average) {
                                success = false;
                                if (brokenCount < maxBrokenCountLog) {
                                    SPDLOG_ERROR(fmt::format(
                                        "Broken Invariant : ForAll "
                                        "l in lightPartition : l <= average."
                                        "[at = {}, l = {}, average = {}, lightCount = {}",
                                        i, resultPartitionLight[i], average, lightCount));
                                }
                                brokenCount += 1;
                            }
                        }
                        if (brokenCount > maxBrokenCountLog) {
                            SPDLOG_ERROR(fmt::format("Broken Invariant : ForAll "
                                                     "l in lightPartition : l <= average."
                                                     "Failed {} more times",
                                                     brokenCount - maxBrokenCountLog));
                        }
                    }
                    // Invariant: ForAll: heavy > average
                    {
                        uint32_t brokenCount = 0;
                        uint32_t maxBrokenCountLog = 4;
                        for (size_t i = 0; i < heavyCount; ++i) {
                            if (resultPartitionHeavy[i] <= average) {
                                success = false;
                                if (brokenCount < maxBrokenCountLog) {
                                    SPDLOG_ERROR(fmt::format(
                                        "Broken Invariant : ForAll "
                                        "h in heavyPartition : h > average."
                                        "[at = {}, h = {}, average = {}, heavyCount = {}",
                                        i, resultPartitionHeavy[i], average, heavyCount));
                                }
                                brokenCount += 1;
                            }
                        }
                        if (brokenCount > maxBrokenCountLog) {
                            SPDLOG_ERROR(fmt::format("Broken Invariant : ForAll "
                                                     "h in heavyPartition : h <= average."
                                                     "Failed {} more times",
                                                     brokenCount - maxBrokenCountLog));
                        }
                    }
                    // Invariant: lightPartitionPrefix is monotone
                    {
                        uint32_t brokenCount = 0;
                        uint32_t maxBrokenCountLog = 4;
                        for (size_t i = 1; i < resultPartitionPrefixLight.size(); ++i) {
                            if (resultPartitionPrefixLight[i] < resultPartitionPrefixLight[i - 1]) {
                                success = false;
                                if (brokenCount < maxBrokenCountLog) {
                                    SPDLOG_ERROR(fmt::format(
                                        "Broken Invariant: \"light "
                                        "partition prefix sum is monotone\"."
                                        "Broken at index = {}. prefix[{}] = {}, prefix[{} - 1] = "
                                        "{}.",
                                        i, i, resultPartitionPrefixLight[i], i,
                                        resultPartitionPrefixLight[i - 1]));
                                }
                                brokenCount += 1;
                            }
                        }
                        if (brokenCount > maxBrokenCountLog) {
                            SPDLOG_ERROR(
                                fmt::format("Broken Invariant: \"light "
                                            "partition prefix sum is monotone\". Broken at {} out "
                                            "of {} indices",
                                            brokenCount, lightCount));
                        }
                    }
                    // Invariant: heavyPartitionPrefix is monotone
                    {
                        uint32_t brokenCount = 0;
                        uint32_t maxBrokenCountLog = 4;
                        for (size_t i = 1; i < resultPartitionPrefixHeavy.size(); ++i) {
                            if (resultPartitionPrefixHeavy[i] < resultPartitionPrefixHeavy[i - 1]) {
                                if (brokenCount < maxBrokenCountLog) {
                                    SPDLOG_ERROR(fmt::format(
                                        "Broken Invariant: \"heavy "
                                        "partition prefix sum is monotone\"."
                                        "Broken at index = {}. prefix[{}] = {}, prefix[{} - 1] = "
                                        "{}.",
                                        i, i, resultPartitionPrefixHeavy[i], i,
                                        resultPartitionPrefixHeavy[i - 1]));
                                }
                                brokenCount += 1;
                            }
                        }
                        if (brokenCount > maxBrokenCountLog) {
                            SPDLOG_ERROR(
                                fmt::format("Broken Invariant: \"heavy "
                                            "partition prefix sum is monotone\". Broken at {} out "
                                            "of {} indices",
                                            brokenCount, heavyCount));
                        }
                    }
                }
                SPDLOG_DEBUG("Compute reference results");
                const auto [d_heavyPartition, d_lightPartition] =
                    wrs::cpu::stable::partition(d_weights, d_average);
                const std::vector<double> d_heavyPartitionPrefix =
                    wrs::cpu::stable::prefix_sum(d_heavyPartition);
                const std::vector<double> d_lightPartitionPrefix =
                    wrs::cpu::stable::prefix_sum(d_lightPartition);
                const auto [heavyPartition, lightPartition] =
                    wrs::cpu::stable::partition(weights, average);
                const std::vector<weight_t> heavyPartitionPrefix =
                    wrs::cpu::stable::prefix_sum<weight_t>(heavyPartition);
                const std::vector<weight_t> lightPartitionPrefix =
                    wrs::cpu::stable::prefix_sum<weight_t>(lightPartition);

                SPDLOG_DEBUG("Compare against reference");
                { // Compare partition with cpu result
                    // Compare heavy count
                    if (heavyPartition.size() != heavyCount) {
                        if (heavyCount > heavyPartition.size()) {
                            success = false;
                            auto minWeightInHeavyIt = std::min_element(resultPartitionHeavy.begin(),
                                                                       resultPartitionHeavy.end());
                            weight_t minWeightInHeavy = *minWeightInHeavyIt;

                            auto minWeightInHeavyRefIt =
                                std::min_element(heavyPartition.begin(), heavyPartition.end());
                            weight_t minWeightInHeavyRef = *minWeightInHeavyRefIt;
                            SPDLOG_ERROR("Heavy count does not match the "
                                         "reference. Expected {}, got {}. minWeightInHeavy = {}, "
                                         "minWeightInRef = {}, pivot = {}",
                                         heavyPartition.size(), heavyCount, minWeightInHeavy,
                                         minWeightInHeavyRef, average);
                        } else {
                            success = false;
                            auto maxWeightInHeavyIt = std::max_element(resultPartitionLight.begin(),
                                                                       resultPartitionLight.end());
                            weight_t maxWeightInHeavy = *maxWeightInHeavyIt;
                            auto maxWeightInHeavyRefIt =
                                std::max_element(lightPartition.begin(), lightPartition.end());
                            weight_t maxWeightInHeavyRef = *maxWeightInHeavyRefIt;
                            SPDLOG_ERROR("Heavy count does not match the "
                                         "reference. Expected {}, got {}. maxWeightInHeavy = {}, "
                                         "maxWeightInRef = {}, pivot = {}",
                                         heavyPartition.size(), heavyCount, maxWeightInHeavy,
                                         maxWeightInHeavyRef, average);
                        }
                    }

                    // Compare light partition.
                    {
                        uint32_t failCount = 0;
                        uint32_t maxFailCountLog = 4;
                        for (size_t i = 0; i < lightCount; ++i) {
                            weight_t cpu = lightPartition[i];
                            weight_t gpu = resultPartitionLight[i];
                            if (cpu != gpu) {
                                success = false;
                                if (failCount < maxFailCountLog) {
                                    SPDLOG_ERROR(fmt::format(
                                        "{}nth element of the light "
                                        "partition"
                                        " does not match the reference. Expected {}, got {}",
                                        i, cpu, gpu));
                                }
                                failCount += 1;
                            }
                        }
                        if (failCount > maxFailCountLog) {
                            SPDLOG_ERROR(fmt::format("Light partition does not match the "
                                                     "reference at {} out of {} indices",
                                                     failCount, lightCount));
                        }
                    }
                    // Compare heavy partition.
                    {
                        uint32_t failCount = 0;
                        uint32_t maxFailCountLog = 4;
                        for (size_t i = 0; i < heavyCount; ++i) {
                            weight_t cpu = heavyPartition[i];
                            weight_t gpu = resultPartitionHeavy[i];
                            if (cpu != gpu) {
                                success = false;
                                if (failCount < maxFailCountLog) {
                                    SPDLOG_ERROR(fmt::format(
                                        "{}nth element of the heavy "
                                        "partition"
                                        " does not match the reference. Expected {}, got {}",
                                        i, cpu, gpu));
                                }
                                failCount += 1;
                            }
                        }
                        if (failCount > maxFailCountLog) {
                            SPDLOG_ERROR(fmt::format("Heavy partition does not match the "
                                                     "reference at {} out {} of indices",
                                                     failCount, heavyCount));
                        }
                    }
                    // Compare light parition prefix sum
                    {
                        uint32_t failCount = 0;
                        uint32_t maxFailCountLog = 4;
                        for (size_t i = 0; i < lightCount; ++i) {
                            weight_t cpu = lightPartitionPrefix[i];
                            weight_t gpu = resultPartitionPrefixLight[i];
                            /* fmt::println("x {} <=> {}   - {}", cpu, gpu, std::abs(cpu / gpu)
                             * - 1.0f); */
                            if (std::abs(std::abs(cpu / gpu) - 1.0f) > 0.001) {
                                success = false;
                                if (failCount < maxFailCountLog) {
                                    SPDLOG_ERROR(
                                        fmt::format("Light partition "
                                                    "prefix sum does not match the reference "
                                                    "at {}. Expected {}, got {}",
                                                    i, cpu, gpu));
                                }
                                failCount += 1;
                            }
                        }
                        if (failCount > maxFailCountLog) {
                            SPDLOG_ERROR(fmt::format("Light partition "
                                                     "prefix sum does not match the reference "
                                                     "at {} out {} of indices",
                                                     failCount, lightCount));
                        }
                    }
                    // Compare heavy parition prefix sum
                    {
                        uint32_t failCount = 0;
                        uint32_t maxFailCountLog = 4;
                        for (size_t i = 0; i < heavyCount; ++i) {
                            weight_t cpu = heavyPartitionPrefix[i];
                            weight_t gpu = resultPartitionPrefixHeavy[i];
                            if (std::abs(std::abs(cpu / gpu) - 1.0f) > 0.001) {
                                if (failCount < maxFailCountLog) {
                                    SPDLOG_ERROR(
                                        fmt::format("Heavy partition "
                                                    "prefix sum does not match the reference"
                                                    "at {}. Expected {}, got {}",
                                                    i, cpu, gpu));
                                }
                                failCount += 1;
                            }
                        }
                        if (failCount > maxFailCountLog) {
                            SPDLOG_ERROR(fmt::format("Heavy partition "
                                                     "prefix sum does not match the reference"
                                                     "at {} out {} of indices",
                                                     failCount, heavyCount));
                        }
                    }
                }

                SPDLOG_DEBUG("Evaluating numerical stability of the prefix sum.");
                {
                    { // Compute worst case inaccuracy of the light partition prefix sum.
                        double worstError = 0.0f;
                        double worstExpected;
                        weight_t worstGot;
                        bool anyError = false;
                        for (size_t i = 0; i < lightCount; ++i) {
                            double cpu = lightPartitionPrefix[i];
                            weight_t gpu = resultPartitionPrefixLight[i];
                            double error = std::abs(cpu - static_cast<double>(gpu));
                            if (error > worstError) {
                                anyError = true;
                                worstError = error;
                                worstExpected = cpu;
                                worstGot = gpu;
                            }
                        }
                        if (worstError > 0.001) {
                            SPDLOG_WARN(
                                fmt::format("Light partition prefix sum is "
                                            "numerically unstable."
                                            "Worst error = {}, with expected {} where value was {}",
                                            worstError, worstExpected, worstGot));
                        } else if (anyError) {
                            SPDLOG_DEBUG(
                                fmt::format("LightPartitionPrefx worst error = {}", worstError));
                            /* SPDLOG_INFO( */
                            /*     fmt::format("Light partition prefix is " */
                            /*                 "kind of stable. Worst error = {}, with " */
                            /*                 "expected {} where value was {}", */
                            /*                 worstError, worstExpected, worstGot)); */

                        } else {
                            SPDLOG_DEBUG(fmt::format("Light partition "
                                                     "prefix is numerically stable"));
                        }
                    }
                    { // Compute worst case inaccuracy of the heavy partition prefix sum.
                        double worstError = 0.0f;
                        double worstExpected = 0;
                        weight_t worstGot = 0;
                        bool anyError = false;
                        for (size_t i = 0; i < heavyCount; ++i) {
                            double cpu = heavyPartitionPrefix[i];
                            weight_t gpu = resultPartitionPrefixHeavy[i];
                            double error = std::abs(cpu - static_cast<double>(gpu));
                            if (error > worstError) {
                                anyError = true;
                                worstError = error;
                                worstExpected = cpu;
                                worstGot = gpu;
                            }
                        }
                        if (worstError > 0.001) {
                            SPDLOG_DEBUG(
                                fmt::format("Heavy partition prefix sum is "
                                            "numerically unstable. "
                                            "Worst error = {}, with expected {} where value was {}",
                                            worstError, worstExpected, worstGot));
                        } else if (anyError) {
                            SPDLOG_DEBUG(
                                fmt::format("HeavyPartitionPrefx worst error = {}", worstError));

                        } else {
                            SPDLOG_DEBUG(fmt::format("PartitionPrefixSumAlgo: heavy partition "
                                                     "prefix is numerically stable"));
                        }
                    }
                }
                SPDLOG_INFO("End testcase");
                configResults.push_back(TestResult{weightGen, success});
            }
            results.push_back(std::make_tuple(algoConfig, configResults));
        }
        SPDLOG_DEBUG("Validation done");
        profiler->collect(true);
        std::cout << merian::Profiler::get_report_str(profiler->get_report()) << std::endl;

        for (const auto& [algoConfig, configResults] : results) {
            fmt::println(
                "Test results for algorithm configuration: [workgroupSize = {}, rows = {}]",
                algoConfig.workgroupSize, algoConfig.rows);
            for (const auto& result : configResults) {
              if (result.succ) {
                fmt::println("\x1B[32mSUCCESS\033[0m : [count = {}, distribution = {}]",
                    result.weightInfo.count, distribution_to_pretty_string(result.weightInfo.distribution));
              }else {
                fmt::println("\x1B[31mFAILURE\033[0m : [count = {}, distribution = {}]",
                    result.weightInfo.count, distribution_to_pretty_string(result.weightInfo.distribution));
              }
            }
        }
    }

  private:
    baseline::DecoupledPartitionAndPrefixSumKernel m_partitionAndPrefixSumKernel;

    merian::BufferHandle m_partitionDescriptorBuffer;
};
} // namespace wrs::baseline
