#include "./BaselineAliasTable.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/cpu/stable.hpp"
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vulkan/vulkan_enums.hpp>

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
