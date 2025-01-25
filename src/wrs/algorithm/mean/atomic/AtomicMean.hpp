#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct AtomicMeanBuffers {
    using Self = AtomicMeanBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle elements;
    using ElementsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    merian::BufferHandle mean;
    using MeanLayout = layout::PrimitiveLayout<float, storageQualifier>;
    using MeanView = layout::BufferView<MeanLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                                    merian::MemoryMappingType memoryMapping,
                                    glsl::uint N) {
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
          buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
              memoryMapping);
          buffers.mean = alloc->createBuffer(MeanLayout::size(),
              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
              memoryMapping);
            
        } else {
          buffers.elements = alloc->createBuffer(ElementsLayout::size(N),
              vk::BufferUsageFlagBits::eTransferSrc,
              memoryMapping);
          buffers.mean = alloc->createBuffer(MeanLayout::size(),
              vk::BufferUsageFlagBits::eTransferDst,
              memoryMapping);

        }
        return buffers;
    }
};

class AtomicMean {
    struct PushConstants {
        glsl::uint N;
    };
  public:
    using Buffers = AtomicMeanBuffers;

    explicit AtomicMean(const merian::ContextHandle& context, glsl::uint workgroupSize, glsl::uint rows) : m_partitionSize(workgroupSize * rows){

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // elements
                .add_binding_storage_buffer() // mean
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/mean/atomic/shader.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(workgroupSize);
        specInfoBuilder.add_entry(rows);
        specInfoBuilder.add_entry(context->physical_device.physical_device_subgroup_properties.subgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, glsl::uint N) {

        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.elements, buffers.mean);
        m_pipeline->push_constant<PushConstants>(cmd, PushConstants{
                                                          .N = N,
        });
        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_partitionSize;
};

}
