#pragma once

#include "src/device/statistics/chi_square/reduce/ChiSquareReduceAllocFlags.hpp"
#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace device {

struct ChiSquareReduceBuffers {
    using Self = ChiSquareReduceBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle histogram;
    using HistogramLayout = host::layout::ArrayLayout<host::glsl::uint64, storageQualifier>;
    using HistogramView = host::layout::BufferView<HistogramLayout>;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle chiSquare;
    using ChiSquareLayout = host::layout::PrimitiveLayout<float, storageQualifier>;
    using ChiSquareView = host::layout::BufferView<ChiSquareLayout>;

    static Self
    allocate(const merian::ResourceAllocatorHandle& alloc,
             merian::MemoryMappingType memoryMapping,
             host::glsl::uint N,
             ChiSquareReduceAllocFlags allocFlags = ChiSquareReduceAllocFlags::ALLOC_ALL) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            if ((allocFlags & ChiSquareReduceAllocFlags::ALLOC_HISTOGRAM) != 0) {
                buffers.histogram = alloc->createBuffer(HistogramLayout::size(N),
                                                        vk::BufferUsageFlagBits::eStorageBuffer |
                                                            vk::BufferUsageFlagBits::eTransferDst,
                                                        memoryMapping);
            }

            if ((allocFlags & ChiSquareReduceAllocFlags::ALLOC_WEIGHTS) != 0) {
                buffers.weights = alloc->createBuffer(WeightsLayout::size(N),
                                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                                          vk::BufferUsageFlagBits::eTransferDst,
                                                      memoryMapping);
            }

            if ((allocFlags & ChiSquareReduceAllocFlags::ALLOC_CHI_SQUARE) != 0) {
                buffers.chiSquare = alloc->createBuffer(ChiSquareLayout::size(),
                                                        vk::BufferUsageFlagBits::eStorageBuffer |
                                                            vk::BufferUsageFlagBits::eTransferSrc,
                                                        memoryMapping);
            }

        } else {

            if ((allocFlags & ChiSquareReduceAllocFlags::ALLOC_HISTOGRAM) != 0) {
                buffers.histogram = alloc->createBuffer(
                    HistogramLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }

            if ((allocFlags & ChiSquareReduceAllocFlags::ALLOC_WEIGHTS) != 0) {
                buffers.weights = alloc->createBuffer(
                    WeightsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            }

            if ((allocFlags & ChiSquareReduceAllocFlags::ALLOC_CHI_SQUARE) != 0) {
                buffers.chiSquare = alloc->createBuffer(
                    ChiSquareLayout::size(), vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
            }
        }
        return buffers;
    }
};

struct ChiSquareReduceConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;

    constexpr ChiSquareReduceConfig() : workgroupSize(512), rows(8) {}
    explicit constexpr ChiSquareReduceConfig(host::glsl::uint workgroupSize, host::glsl::uint rows)
        : workgroupSize(workgroupSize), rows(rows) {}

    constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows;
    }
};

class ChiSquareReduce {
    struct PushConstants {
        host::glsl::uint N;
        host::glsl::uint S;
        host::glsl::f32 totalWeight;
    };

  public:
    using Buffers = ChiSquareReduceBuffers;
    using Config = ChiSquareReduceConfig;

    explicit ChiSquareReduce(const merian::ContextHandle& context,
                             const merian::ShaderCompilerHandle& shaderCompiler,
                             ChiSquareReduceConfig config = {})
        : m_blockSize(config.blockSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // reductions
                .add_binding_storage_buffer() // prefix sum
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/device/statistics/chi_square/reduce/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {"src/device/common/"});

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(config.workgroupSize);
        specInfoBuilder.add_entry(config.rows);
        specInfoBuilder.add_entry(
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             host::glsl::uint S,
             host::glsl::f32 totalWeight) const {

        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.histogram, buffers.weights, buffers.chiSquare);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{
                                                          .N = N,
                                                          .S = S,
                                                          .totalWeight = totalWeight,
                                                      });
        const uint32_t workgroupCount = (N + m_blockSize - 1) / m_blockSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline host::glsl::uint blockSize() const {
        return m_blockSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_blockSize;
};

} // namespace device
