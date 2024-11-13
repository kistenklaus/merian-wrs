#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"
#include "src/wrs/algo/Reduce.hpp"
#include <iostream>
#include <vulkan/vulkan_enums.hpp>

namespace wrs {

class BaselineAliasTable {
    using weight_t = float;

  public:
    BaselineAliasTable(merian::ContextHandle context, uint32_t maxWeights)
        : m_context(context), m_maxWeights(maxWeights), m_reduceAlgo(context) {
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
                m_maxWeights * sizeof(float), vk::BufferUsageFlagBits::eTransferSrc,
                merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);

            m_reducePongBuffer = m_alloc->createBuffer(
                m_reduceAlgo.requiredResultBufferSize(m_maxWeights),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                merian::MemoryMappingType::NONE);
            m_resultBufferStage =
                m_alloc->createBuffer(sizeof(float), vk::BufferUsageFlagBits::eTransferDst,
                                      merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);
        }
    }

    void build(vk::CommandBuffer cmd,
               std::optional<merian::ProfilerHandle> profiler = std::nullopt) {
        m_resultBuffer =
            m_reduceAlgo.run(cmd, m_weightBuffer, m_reducePongBuffer, std::nullopt, profiler);
    }

    void set_weights(vk::CommandBuffer cmd, vk::ArrayProxy<weight_t> weights) {
        assert(weights.size() < m_maxWeights);
        void* mapped = m_weightBufferStage->get_memory()->map();
        std::memcpy(mapped, weights.data(), weights.size() * sizeof(float));
        m_weightBufferStage->get_memory()->unmap();

        vk::BufferCopy copy{0, 0, weights.size() * sizeof(float)};
        cmd.copyBuffer(*m_weightBufferStage, *m_weightBuffer, 1, &copy);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {},
                            m_weightBuffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                           vk::AccessFlagBits::eShaderRead),
                            {});
    }

    const merian::BufferHandle download_result(vk::CommandBuffer cmd) {
        assert(m_resultBuffer != nullptr);
        vk::BufferCopy copy{0, 0, sizeof(float)};
        cmd.copyBuffer(*m_resultBuffer, *m_resultBufferStage, 1, &copy);
        return m_resultBufferStage;
    }

  private:
    const merian::ContextHandle m_context;
    const uint32_t m_maxWeights;

    merian::ResourceAllocatorHandle m_alloc;

    merian::BufferHandle m_weightBuffer;
    merian::BufferHandle m_weightBufferStage;
    merian::BufferHandle m_reducePongBuffer;

    merian::BufferHandle m_resultBuffer = nullptr;
    merian::BufferHandle m_resultBufferStage;

    Reduce m_reduceAlgo;
};

} // namespace wrs
