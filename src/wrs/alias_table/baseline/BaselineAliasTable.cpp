#include "./BaselineAliasTable.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/cpu/stable.hpp"
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>

void wrs::baseline::BaselineAliasTable::cpuValidation(merian::QueueHandle& queue,
                                                      merian::CommandPool& cmdPool,
                                                      uint32_t weightCount) {
    queue->wait_idle();
    { // Download weights
        vk::CommandBuffer cmd = cmdPool.create_and_begin();
        vk::BufferCopy copy{0, 0, weightCount * sizeof(float)};
        cmd.copyBuffer(*m_weightBuffer, *m_resultBufferStage, 1, &copy);
        cmd.end();
        queue->submit_wait(cmd);
    }
    float* weightsMapped = m_resultBufferStage->get_memory()->map_as<float>();
    std::vector<float> gpuWeights(weightCount);
    std::memcpy(gpuWeights.data(), weightsMapped, weightCount * sizeof(float));

    // Download avgPrefixSum
    {
        vk::CommandBuffer cmd = cmdPool.create_and_begin();
        vk::BufferCopy copy{0, 0, weightCount * sizeof(float) + sizeof(float)};
        cmd.copyBuffer(*m_avgPrefixSum, *m_resultBufferStage, 1, &copy);
        cmd.end();
        queue->submit_wait(cmd);
    }
    float* avgPrefixMapped = m_resultBufferStage->get_memory()->map_as<float>();
    float gpuAverage = *avgPrefixMapped;
    std::vector<float> gpuPrefix(weightCount);
    std::memcpy(gpuPrefix.data(), avgPrefixMapped + 1, sizeof(float) * weightCount);
    m_resultBufferStage->get_memory()->unmap();

    SPDLOG_INFO(fmt::format("AliasTable: weightSum  (gpu): {}", gpuPrefix.back()));
    SPDLOG_INFO(fmt::format("AliasTable: weightAvg  (gpu): {}", gpuAverage));

    // Download heavyLight partition
    {
        vk::CommandBuffer cmd = cmdPool.create_and_begin();
        vk::BufferCopy copy{0, 0, m_heavyLight->get_size()};
        cmd.copyBuffer(*m_heavyLight, *m_resultBufferStage, 1, &copy);
        cmd.end();
        queue->submit_wait(cmd);
    }
    void* lightHeavyMapped = m_resultBufferStage->get_memory()->map();
    uint32_t gpuHeavyCount = *reinterpret_cast<uint32_t*>(lightHeavyMapped);
    SPDLOG_INFO(fmt::format("AliasTable: heavyCount (gpu): {}", gpuHeavyCount));
    SPDLOG_INFO(fmt::format("AliasTable: lightCount (gpu): {}", weightCount - gpuHeavyCount));
    std::vector<float> gpuHeavyPartition(gpuHeavyCount);
    std::vector<float> gpuLightPartition(weightCount - gpuHeavyCount);
    std::memcpy(gpuHeavyPartition.data(),
                reinterpret_cast<uint8_t*>(lightHeavyMapped) + sizeof(uint32_t),
                gpuHeavyCount * sizeof(float));
    std::memcpy(gpuLightPartition.data(),
                reinterpret_cast<uint8_t*>(lightHeavyMapped) + sizeof(uint32_t) +
                    gpuHeavyCount * sizeof(float),
                (weightCount - gpuHeavyCount) * sizeof(float));
    std::reverse(gpuLightPartition.begin(), gpuLightPartition.end());
    m_resultBufferStage->get_memory()->unmap();
    // Download heavyLightPrefix
    {
        vk::CommandBuffer cmd = cmdPool.create_and_begin();
        vk::BufferCopy copy{0, 0, weightCount * sizeof(float)};
        cmd.copyBuffer(*m_heavyLightPrefix, *m_resultBufferStage, 1, &copy);
        cmd.end();
        queue->submit_wait(cmd);
    }
    float* lightHeavyPrefixMapped = m_resultBufferStage->get_memory()->map_as<float>();
    std::vector<float> gpuHeavyPrefix(gpuHeavyCount);
    std::vector<float> gpuLightPrefix(weightCount - gpuHeavyCount);
    std::memcpy(gpuHeavyPrefix.data(), lightHeavyPrefixMapped, gpuHeavyCount * sizeof(float));
    std::memcpy(gpuLightPrefix.data(), lightHeavyPrefixMapped + gpuHeavyCount,
                (weightCount - gpuHeavyCount) * sizeof(float));
    std::reverse(gpuLightPrefix.begin(), gpuLightPrefix.end());
    m_resultBufferStage->get_memory()->unmap();

    // Download splitDescriptors
    {
        vk::CommandBuffer cmd = cmdPool.create_and_begin();
        vk::BufferCopy copy{0, 0, m_splitDescriptors->get_size()};
        cmd.copyBuffer(*m_splitDescriptors, *m_resultBufferStage, 1, &copy);
        cmd.end();
        queue->submit_wait(cmd);
    }

    struct __attribute__((packed)) SplitDescriptor {
        uint32_t heavyCount;
        uint32_t lightCount;
        float spill;
    };
    static_assert(sizeof(SplitDescriptor) == (2 * sizeof(uint32_t) + sizeof(float)));

    SplitDescriptor* splitDescriptorsMapped =
        m_resultBufferStage->get_memory()->map_as<SplitDescriptor>();
    std::vector<SplitDescriptor> gpuSplitDescriptors{m_splitDescriptors->get_size() /
                                                     sizeof(SplitDescriptor)};
    std::memcpy(gpuSplitDescriptors.data(), splitDescriptorsMapped,
                gpuSplitDescriptors.size() * sizeof(SplitDescriptor));
    m_resultBufferStage->get_memory()->unmap();

    // Compute (CPU) prefix and average
    std::vector<float> cpuPrefix = wrs::cpu::stable::prefix_sum(gpuWeights);
    double cpuAverage = cpuPrefix[weightCount - 1] / weightCount;
    // Validate prefix and average
    if (std::abs(cpuAverage / gpuAverage) - 1.0 > 0.0001) {
        SPDLOG_ERROR(
            fmt::format("AliasTable:\nValidation failed invalid average. CPU = {}, GPU = {}",
                        cpuAverage, gpuAverage));
    }
    for (size_t i = 0; i < weightCount; ++i) {
        double cpu = cpuPrefix[i];
        float gpu = gpuPrefix[i];
        if (std::abs(cpu / gpu) - 1.0 > 0.0001) {
            SPDLOG_ERROR(fmt::format(
                "AliasTable:\nValidation failed invalid prefix at index {}. CPU = {}, GPU = {}", i,
                cpu, gpu));
        }
    }
    // Compute (CPU) Light Heavy partitions
    auto [cpuHeavy, cpuLight] = wrs::cpu::stable::partition(gpuWeights, gpuAverage);
    // Validate (CPU) Light & Heavy partitions
    if (cpuHeavy.size() != gpuHeavyCount) {
        SPDLOG_ERROR(fmt::format("AliasTable:\nValidation failed invalid partition. "
                                 "Invalid heavy count. CPU = {}, GPU = {}",
                                 cpuHeavy.size(), gpuHeavyCount));
    }
    assert(cpuLight.size() == gpuLightPartition.size());
    for (size_t i = 0; i < cpuHeavy.size(); ++i) {
        if (cpuHeavy[i] != gpuHeavyPartition[i]) {
            SPDLOG_ERROR(
                fmt::format("AliasTable:\nValidation failed invalid heavy partition. Not stable "
                            "or just wrong. CPU = {}, GPU = {}",
                            cpuHeavy[i], gpuHeavyPartition[i]));
        }
    }
    for (size_t i = 0; i < cpuLight.size(); ++i) {
        if (cpuLight[i] != gpuLightPartition[i]) {
            SPDLOG_ERROR(
                fmt::format("AliasTable:\nValidation failed invalid light partition. Not stable "
                            "or just wrong. CPU = {}, GPU = {}",
                            cpuLight[i], gpuLightPartition[i]));
        }
    }
    // Compute (CPU) Light Heavy Prefix Sum
    std::vector<float> cpuHeavyPrefix = wrs::cpu::stable::prefix_sum(cpuHeavy);
    std::vector<float> cpuLightPrefix = wrs::cpu::stable::prefix_sum(cpuLight);
    // Validate (CPU) Light & Heavy Prefix Sum
    for (size_t i = 0; i < cpuHeavyPrefix.size(); ++i) {
        double cpu = cpuHeavyPrefix[i];
        float gpu = gpuHeavyPrefix[i];
        if (std::abs(cpu / gpu) - 1.0 > 0.0001) {
            SPDLOG_ERROR(fmt::format("AliasTable:\nValidation failed invalid heavy "
                                     "prefix at index {}. CPU = {}, GPU = {}",
                                     i, cpu, gpu));
        }
    }
    for (size_t i = 0; i < cpuLightPrefix.size(); ++i) {
        double cpu = cpuLightPrefix[i];
        float gpu = gpuLightPrefix[i];
        if (std::abs(cpu / gpu) - 1.0 > 0.0001) {
            SPDLOG_ERROR(fmt::format("AliasTable:\nValidation failed invalid light "
                                     "prefix at index {}. CPU = {}, GPU = {}",
                                     i, cpu, gpu));
        }
    }

    // Compute (CPU) split

    std::cout << "CPU:" << std::endl;
    const int N = weightCount;
    const int K = splitCount;
    const int K_ = K - 1;
    std::vector<SplitDescriptor> cpuSplitDescriptors;
    cpuSplitDescriptors.push_back(SplitDescriptor{0, 0, 0});
    size_t warnCpuSplitCount = 0;
    constexpr size_t maxCpuWarnSplit = 4;

    /*for (size_t i = 0; i < gpuLightPrefix.size(); ++i) {*/
    /*  std::cout << "[i] = " << gpuLightPrefix[i] << std::endl;*/
    /*}*/

    for (size_t k = 1; k < K - 1; k++) {
        const int n_ = N * k;
        const int n = int((n_ + K_ - 1) / K_);
        auto info = wrs::cpu::stable::split(cpuHeavy, cpuHeavyPrefix, cpuLightPrefix,
                                            (float)cpuAverage, n);
        std::cout << "CPU Split : {" << info.heavyCount << ", " << info.lightCount << ", " << info.spill << "}"
                  << std::endl;
        if (info.heavyCount + info.lightCount != n && warnCpuSplitCount < maxCpuWarnSplit) {
            warnCpuSplitCount += 1;
            SPDLOG_WARN("AliasTable:\nInvalid cpu split: invariant i + j != n broken");
        }
        float sigma = gpuHeavyPrefix[info.heavyCount - 1] + gpuLightPrefix[info.lightCount - 1];
        float localW = gpuAverage * n;
        if (!(sigma <= localW) && warnCpuSplitCount < maxCpuWarnSplit) {
            warnCpuSplitCount += 1;
            SPDLOG_WARN(fmt::format("AliasTable:\nInvalid cpu splt: invariant sigma <= localW "
                                    "broken. sigma = {}, localW = {}",
                                    sigma, localW));
        }
        if (info.spill < 0 && warnCpuSplitCount < maxCpuWarnSplit) {
            warnCpuSplitCount += 1;
            SPDLOG_WARN(
                "AliasTable:\nInvaliad cpu split: invariant sigma + h[j+1] > localW broke brokenn");
        }
        cpuSplitDescriptors.push_back(SplitDescriptor{static_cast<uint32_t>(info.heavyCount),
                                                      static_cast<uint32_t>(info.lightCount),
                                                      info.spill});
    }
    cpuSplitDescriptors.push_back(
        SplitDescriptor{static_cast<uint32_t>(N), static_cast<uint32_t>(N), 0});

    // Validate split descriptors
    if (cpuSplitDescriptors.size() != gpuSplitDescriptors.size()) {
        SPDLOG_ERROR("AliasTable:\nValidation failed gpu generated a different amount of gpu split "
                     "descriptors. Probably a bug in the validation. Skipping split validation.");
    } else {
        size_t warnSplitCounter = 0;
        const size_t maxWarnSplit = 4;

        size_t errorSplitCounter = 0;
        const size_t maxErrSplit = 4;

        for (size_t i = 0; i < cpuSplitDescriptors.size(); ++i) {
            const auto& cpu = cpuSplitDescriptors[i];
            const auto& gpu = gpuSplitDescriptors[i];

            // Warnings
            if (cpu.heavyCount != gpu.heavyCount && warnSplitCounter < maxWarnSplit) {
                warnSplitCounter += 1;
                SPDLOG_WARN(
                    fmt::format("AliasTable:\nsplit implementation is not equivalent to "
                                "reference. Expected SplitDescriptor.heavyCount = {}, got {}",
                                cpu.heavyCount, gpu.heavyCount));
            }
            if (cpu.lightCount != gpu.lightCount && warnSplitCounter < maxWarnSplit) {
                warnSplitCounter += 1;
                SPDLOG_WARN(
                    fmt::format("AliasTable:\nsplit implementation is not equivalent to "
                                "reference. Expected SplitDescriptor.lightCount = {}, got {}",
                                cpu.lightCount, gpu.lightCount));
            }

            if (std::abs(cpu.spill - gpu.spill) > 0.01 && warnSplitCounter < maxWarnSplit) {
                warnSplitCounter += 1;
                SPDLOG_WARN(fmt::format("AliasTable:\nsplit implementation is not equivalent to "
                                        "reference. Expected SplitDescriptor.spill = {}, got {}",
                                        cpu.spill, gpu.spill));
            }

            // Errors
            if (i == 0) {
                if (gpu.heavyCount != 0 && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(fmt::format("AliasTable:\nfirst split descriptor has invalid "
                                             "heavyCount. Expected {}, got {}",
                                             0, gpu.heavyCount));
                }
                if (gpu.lightCount != 0 && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(fmt::format("AliasTable:\nfirst split descriptor has invalid "
                                             "lightCount. Expected {}, got {}",
                                             0, gpu.lightCount));
                }
                if (std::abs(gpu.spill) > 0.001 && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(fmt::format("AliasTable:\nfirst split descriptor has invalid "
                                             "spill. Expected {}, got {}",
                                             0, gpu.spill));
                }
            } else if (i == cpuSplitDescriptors.size() - 1 && errorSplitCounter < maxErrSplit) {
                if (gpu.heavyCount != static_cast<uint32_t>(N)) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(fmt::format("AliasTable:\nlast split descriptor has invalid "
                                             "heavyCount. Expected {}, got {}",
                                             N, gpu.heavyCount));
                }
                if (gpu.lightCount != static_cast<uint32_t>(N) && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(fmt::format("AliasTable:\nlast split descriptor has invalid "
                                             "lightCount. Expected {}, got {}",
                                             N, gpu.lightCount));
                }
                if (std::abs(gpu.spill) > 0.0001 && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(fmt::format(
                        "AliasTable:\nlast split descriptor has invalid spill. Expected {}, got {}",
                        0.0f, gpu.spill));
                }
            } else {
                const int n_ = N * i;
                const int n = int((n_ + K_ - 1) / K_);
                if (gpu.heavyCount + gpu.lightCount != static_cast<uint32_t>(n) &&
                    errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(
                        fmt::format("AliasTable:\nInvariant: heavyCount + lightCount = n is "
                                    "broken. [{} + {} != {}]",
                                    gpu.heavyCount, gpu.lightCount, n));
                }

                if ((gpu.heavyCount > gpuHeavyCount && gpu.heavyCount != 0) &&
                    errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(
                        fmt::format("AliasTable:\n Invalid split descriptor. Heavy count."));
                }
                if (((gpu.lightCount > weightCount - gpuHeavyCount) && gpu.lightCount != 0) &&
                    errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(
                        fmt::format("AliasTable:\n Invalid split descriptor. Light count."));
                }
                float sigma = 0.0f;
                if (gpu.heavyCount > 0) {
                    sigma += gpuHeavyPrefix[gpu.heavyCount - 1];
                }
                if (gpu.lightCount > 0) {
                    sigma += gpuLightPrefix[gpu.lightCount - 1];
                }
                float localW = gpuAverage * n;
                if (sigma > localW && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(
                        fmt::format("AliasTable:\nInvariant: sigma < localW is broken. [{} !< {}]",
                                    sigma, localW));
                }

                if (gpu.spill < 0 && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR("AliasTable:\ninvalid spill. The spill is always expected to be "
                                 "position. If not it might indicate a broken invariant.");
                }

                float nextHeavy = gpuHeavyPartition[gpu.heavyCount];
                if (sigma + nextHeavy <= localW && errorSplitCounter < maxErrSplit) {
                    errorSplitCounter += 1;
                    SPDLOG_ERROR(
                        fmt::format("AliasTable:\nInvariant: sigma + heavy[j+1] <= localW is "
                                    "broken. [{} + {} !<= {}]",
                                    sigma, nextHeavy, localW));
                }
            }
        }
    }
}

const std::tuple<merian::BufferHandle, uint32_t>
wrs::baseline::BaselineAliasTable::download_result(vk::CommandBuffer cmd,
                                                   std::optional<merian::ProfilerHandle> profiler) {
#ifdef MERIAN_PROFILER_ENABLE
    std::optional<merian::ProfileScopeGPU> merian_profiler_scope = std::nullopt;
    if (profiler.has_value()) {
        merian_profiler_scope.emplace(profiler.value(), cmd, "download result");
    }
#endif
    assert(m_resultBuffer != nullptr);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer, {}, {},
                        m_resultBuffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                       vk::AccessFlagBits::eTransferRead),
                        {});

    vk::BufferCopy copy{0, 0, m_resultBuffer->get_size()};
    cmd.copyBuffer(*m_resultBuffer, *m_resultBufferStage, 1, &copy);
    return {m_resultBufferStage, m_resultCount};
}

void wrs::baseline::BaselineAliasTable::set_weights(
    vk::CommandBuffer cmd,
    vk::ArrayProxy<weight_t> weights,
    std::optional<merian::ProfilerHandle> profiler) {
    assert(weights.size() < m_maxWeights);
#ifdef MERIAN_PROFILER_ENABLE
    std::optional<merian::ProfileScopeGPU> merian_profiler_scope = std::nullopt;
    if (profiler.has_value()) {
        merian_profiler_scope.emplace(
            profiler.value(), cmd,
            fmt::format("update weights [weightCount = {}]", weights.size()));
    }
#endif

    std::cout << "count: " << weights.size() << std::endl;
    void* mapped = m_weightBufferStage->get_memory()->map();
    std::memcpy(mapped, weights.data(), weights.size() * sizeof(weight_t));
    m_weightBufferStage->get_memory()->unmap();

    vk::BufferCopy copy{0, 0, weights.size() * sizeof(weight_t)};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer, {},
                        {},
                        m_weightBufferStage->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                            vk::AccessFlagBits::eTransferRead),
                        {});
    cmd.copyBuffer(*m_weightBufferStage, *m_weightBuffer, 1, &copy);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        m_weightBuffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                       vk::AccessFlagBits::eShaderRead),
                        {});
}

wrs::baseline::BaselineAliasTable::BaselineAliasTable(merian::ContextHandle context,
                                                      uint32_t maxWeights)
    : m_context(context), m_maxWeights(maxWeights), m_prefixSumAndAverage(context, maxWeights),
      m_partitionAndPrefixSum(context, maxWeights), m_split(context) {
    { // Fetch ResouceAllocator
        auto resources = m_context->get_extension<merian::ExtensionResources>();
        assert(resources != nullptr);
        m_alloc = resources->resource_allocator();
    }

    { // Create buffers
        m_weightBuffer = m_alloc->createBuffer(m_maxWeights * sizeof(float),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst |
                                                   vk::BufferUsageFlagBits::eTransferSrc,
                                               merian::MemoryMappingType::NONE);

        m_weightBufferStage = m_alloc->createBuffer(
            m_maxWeights * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc,
            merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);

        m_heavyLight = m_alloc->createBuffer(sizeof(unsigned int) + m_maxWeights * sizeof(float),
                                             vk::BufferUsageFlagBits::eStorageBuffer |
                                                 vk::BufferUsageFlagBits::eTransferSrc,
                                             merian::MemoryMappingType::NONE);
        m_heavyLightPrefix = m_alloc->createBuffer(m_maxWeights * sizeof(float),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferSrc,
                                                   merian::MemoryMappingType::NONE);

        m_avgPrefixSum = m_alloc->createBuffer(sizeof(weight_t) + m_maxWeights * sizeof(weight_t),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferSrc,
                                               merian::MemoryMappingType::NONE);
        m_splitDescriptors = m_alloc->createBuffer(splitCount * m_split.splitDescriptorSize(),
                                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                                       vk::BufferUsageFlagBits::eTransferSrc,
                                                   merian::MemoryMappingType::NONE);

        m_resultBufferStage = m_alloc->createBuffer(
            m_maxWeights * sizeof(float) * 2, vk::BufferUsageFlagBits::eTransferDst,
            merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);
    }
}

void wrs::baseline::BaselineAliasTable::build(vk::CommandBuffer cmd,
                                              std::optional<merian::ProfilerHandle> profiler) {
#ifdef MERIAN_PROFILER_ENABLE
    std::optional<merian::ProfileScopeGPU> merian_profiler_scope = std::nullopt;
    if (profiler.has_value()) {
        merian_profiler_scope.emplace(profiler.value(), cmd, "build alias table");
    }
#endif
    uint32_t N = m_weightBuffer->get_size() / sizeof(weight_t);

    m_prefixSumAndAverage.run(cmd, m_weightBuffer, m_avgPrefixSum, N, profiler);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        m_avgPrefixSum->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                       vk::AccessFlagBits::eShaderRead),
                        {});

    m_partitionAndPrefixSum.run(cmd, m_weightBuffer, m_avgPrefixSum, m_heavyLight,
                                m_heavyLightPrefix, N, profiler);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        {m_heavyLight->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eShaderRead),
                         m_heavyLightPrefix->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eShaderRead)},
                        {});
    /* m_split.run(cmd, m_weightBuffer, m_avgPrefixSum, m_heavyLight, m_heavyLightPrefix, */
    /*             m_splitDescriptors, splitCount, N, profiler); */

    m_resultBuffer = m_splitDescriptors;
    m_resultCount = splitCount;
}
