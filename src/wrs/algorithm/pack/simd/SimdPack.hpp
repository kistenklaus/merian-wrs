#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "src/wrs/layout/Attribute.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"
namespace wrs {

struct SimdPackBuffers {
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    using weight_type = glsl::float_t;

    merian::BufferHandle partitionIndices; // bind = 0
    using PartitionIndicesLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "heavyCount">,
                             layout::Attribute<glsl::uint*, "heavyLightIndices">>;
    using PartitionIndicesView = layout::BufferView<PartitionIndicesLayout>;

    merian::BufferHandle weights; // binding = 1
    using WeightsLayout = layout::ArrayLayout<weight_type, storageQualifier>;
    using WeightsView = layout::BufferView<WeightsLayout>;

    merian::BufferHandle mean; // binding = 2
    using MeanLayout = layout::PrimitiveLayout<weight_type, storageQualifier>;
    using MeanView = layout::BufferView<MeanLayout>;

    merian::BufferHandle splits; // binding = 3
    using SplitStructLayout = layout::StructLayout<storageQualifier,
                                                   layout::Attribute<glsl::uint, "i">,
                                                   layout::Attribute<glsl::uint, "j">,
                                                   layout::Attribute<weight_type, "spill">>;
    using SplitsLayout = layout::ArrayLayout<SplitStructLayout, storageQualifier>;
    using SplitsView = layout::BufferView<SplitsLayout>;

    merian::BufferHandle aliasTable; // binding = 4
    using AliasTableEntryLayout = layout::StructLayout<storageQualifier,
                                                       layout::Attribute<weight_type, "p">,
                                                       layout::Attribute<glsl::uint, "a">>;
    using AliasTableLayout = layout::ArrayLayout<AliasTableEntryLayout, storageQualifier>;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    static SimdPackBuffers allocate(const merian::ResourceAllocatorHandle& alloc,
                                    const std::size_t weightCount,
                                    const std::size_t splitCount,
                                    merian::MemoryMappingType memoryMapping) {
        SimdPackBuffers buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
            buffers.partitionIndices = alloc->createBuffer(
                PartitionIndicesLayout::size(weightCount),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                memoryMapping);
            buffers.weights = alloc->createBuffer(WeightsLayout::size(weightCount),
                                                  vk::BufferUsageFlagBits::eStorageBuffer |
                                                      vk::BufferUsageFlagBits::eTransferDst,
                                                  memoryMapping);
            buffers.mean = alloc->createBuffer(MeanLayout::size(),
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               memoryMapping);
            buffers.splits = alloc->createBuffer(SplitsLayout::size(splitCount),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferDst,
                                                 memoryMapping);
            buffers.aliasTable = alloc->createBuffer(AliasTableLayout::size(weightCount),
                                                     vk::BufferUsageFlagBits::eStorageBuffer |
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                     memoryMapping);
        } else {
            buffers.partitionIndices =
                alloc->createBuffer(PartitionIndicesLayout::size(weightCount),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.weights =
                alloc->createBuffer(WeightsLayout::size(weightCount),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.mean = alloc->createBuffer(
                MeanLayout::size(), vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.splits =
                alloc->createBuffer(SplitsLayout::size(splitCount),
                                    vk::BufferUsageFlagBits::eTransferSrc, memoryMapping);
            buffers.aliasTable =
                alloc->createBuffer(AliasTableLayout::size(weightCount),
                                    vk::BufferUsageFlagBits::eTransferDst, memoryMapping);
        }
        return buffers;
    }
};

template <typename T> class SimdPack {
    static_assert(std::same_as<T, float>, "Other weights are currently not supported");
    struct PushConstant {
        glsl::uint N;
        glsl::uint K;
    };
    static_assert(std::is_standard_layout_v<PushConstant>);
    static_assert(std::is_trivial_v<PushConstant>);

  public:
    using Buffers = SimdPackBuffers;
    using weight_t = T;

    explicit SimdPack(const merian::ContextHandle& context, const glsl::uint workgroupSize = 32) : m_workgroupSize(workgroupSize) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/pack/simd/float.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstant>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);

        m_writes.resize(5);

        vk::WriteDescriptorSet& partitionIndices = m_writes[0];
        partitionIndices.setDstBinding(0);
        partitionIndices.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& weights = m_writes[1];
        weights.setDstBinding(1);
        weights.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& mean = m_writes[2];
        mean.setDstBinding(2);
        mean.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& splits = m_writes[3];
        splits.setDstBinding(3);
        splits.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& table = m_writes[4];
        table.setDstBinding(4);
        table.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    }

    void run(const vk::CommandBuffer cmd, glsl::uint N, glsl::uint K, const Buffers& buffers) {

        m_pipeline->bind(cmd);

        vk::DescriptorBufferInfo partitionIndicesDesc =
            buffers.partitionIndices->get_descriptor_info();
        m_writes[0].setBufferInfo(partitionIndicesDesc);
        vk::DescriptorBufferInfo weightsDesc = buffers.weights->get_descriptor_info();
        m_writes[1].setBufferInfo(weightsDesc);
        vk::DescriptorBufferInfo meanDesc = buffers.mean->get_descriptor_info();
        m_writes[2].setBufferInfo(meanDesc);
        vk::DescriptorBufferInfo splitsDesc = buffers.splits->get_descriptor_info();
        m_writes[3].setBufferInfo(splitsDesc);
        vk::DescriptorBufferInfo aliasTableDesc = buffers.aliasTable->get_descriptor_info();
        m_writes[4].setBufferInfo(aliasTableDesc);
        m_pipeline->push_descriptor_set(cmd, m_writes);

        m_pipeline->push_constant<PushConstant>(cmd, PushConstant{.N = N, .K = K});
        const uint32_t workgroupCount = (K + m_workgroupSize - 1) / m_workgroupSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
    std::vector<vk::WriteDescriptorSet> m_writes;
};

} // namespace wrs
