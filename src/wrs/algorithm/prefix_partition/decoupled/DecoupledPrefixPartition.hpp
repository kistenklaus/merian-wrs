#pragma once

#include "merian/vk/memory/resource_allocations.hpp"

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/PrimitiveLayout.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct DecoupledPrefixPartitionBuffers {
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;
    using element_type = glsl::f32;
    /**
     * Buffer, which contains all elements
     * over which the partition and the
     * prefix sums are computed.
     * The buffer is requiring to have a packed layout!.
     *
     * Layout : T[[N]]
     *
     * MinSize: sizeof(T) * N
     */
    merian::BufferHandle elements;
    static constexpr auto ELEMENT_BUFFER_USAGE_FLAGS = vk::BufferUsageFlagBits::eStorageBuffer;
    using ElementsLayout = layout::ArrayLayout<element_type, storageQualifier>;
    using ElementsView = layout::BufferView<ElementsLayout>;

    /**
     * Buffer, which contains the pivot element.
     *
     * Layout :  T[[0]] = pivot
     *
     * MinSize: sizeof(T)
     */
    merian::BufferHandle pivot;
    static constexpr auto PIVOT_BUFFER_USAGE_FLAGS = vk::BufferUsageFlagBits::eStorageBuffer;
    using PivotLayout = layout::PrimitiveLayout<float, storageQualifier>;
    using PivotView = layout::BufferView<PivotLayout>;

    /**
     * Buffer, which holds internal state of the decoupled lookback.
     * Before a kernel is executed it is required that this buffer is zeroed!
     *
     * Layout : implementation defined.
     * MinSize: ((DecoupledPrefixPartitionKernel)instance).minBufferDescriptorSize(N)
     */
    merian::BufferHandle batchDescriptors;
    static constexpr auto BATCH_DESCRIPTOR_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;

    /* struct BatchDescriptor { */
    /*     uint heavyCount; */
    /*     uint heavyCountInclusivePrefix; */
    /*     monoid heavySum; */
    /*     monoid heavyInclusivePrefix; */
    /*     monoid lightSum; */
    /*     monoid lightInclusivePrefix; */
    /*     state_t state; */
    /* }; */
    using _BatchDescriptorLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "heavyCount">,
                             layout::Attribute<glsl::uint, "heavyCountInclusivePrefix">,
                             layout::Attribute<element_type, "heavySum">,
                             layout::Attribute<element_type, "heavyInclusivePrefix">,
                             layout::Attribute<element_type, "lightSum">,
                             layout::Attribute<element_type, "lightInclusivePrefix">,
                             layout::Attribute<glsl::uint, "state">>;

    using _BatchDescriptorArrayLayout =
        layout::ArrayLayout<_BatchDescriptorLayout, storageQualifier>;

    using BatchDescriptorsLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<glsl::uint, "atomicBatchCounter">,
                             layout::Attribute<_BatchDescriptorArrayLayout, "batchInfo">>;
    using BatchDescriptorsView = layout::BufferView<BatchDescriptorsLayout>;

    constexpr static vk::DeviceSize minBatchDescriptorSize(const uint32_t N,
                                                           const uint32_t partitionSize,
                                                           const size_t sizeof_weight) {
        const uint32_t workgroupCount = (N + partitionSize - 1) / partitionSize;
        vk::DeviceSize batchDescriptorSize = 4 * sizeof_weight + 2 * sizeof_weight + sizeof_weight;
        // TODO consider proper padding this is just a upper bound for a guess!
        batchDescriptorSize += sizeof(uint32_t) * 4;
        return batchDescriptorSize * workgroupCount;
    }

    constexpr static vk::DeviceSize minBatchDescriptorSize(const uint32_t N,
                                                           const uint32_t workgroupSize,
                                                           const uint32_t rows,
                                                           const size_t sizeOfWeight) {
        const uint32_t partitionSize = workgroupSize * rows;
        return minBatchDescriptorSize(N, partitionSize, sizeOfWeight);
    }

    /**
     * Buffer, which contains the prefix over the partitions.
     *
     * Layout : T[[N + 1]]
     * The first element contains amount of element n,
     * which are above the pivot element.
     * Elements 1,...n contains the prefix over all elements above the pivot
     * Elements n+1,...,N+1 contain the prefix over all elements
     * below or equal to the pivot.
     *
     * MinSize: sizeof(T) * (N+1)
     */
    merian::BufferHandle partitionPrefix;
    static constexpr vk::BufferUsageFlagBits PREFIX_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    static constexpr vk::DeviceSize minPartitionPrefixSize(std::size_t N,
                                                           vk::DeviceSize sizeOfElement) {
        return sizeOfElement * N + sizeof(glsl::uint);
    }
    using PartitionPrefixLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<wrs::glsl::uint, "heavyCount">,
                             layout::Attribute<element_type*, "heavyLightPrefix">>;
    using PartitionPrefixView = layout::BufferView<PartitionPrefixLayout>;

    /**
     * Buffer, which contains both partitions.
     * Must be set iff. WRITE_PARTITION is set.
     *
     * Layout: T[[N]]
     * Elements 0,...n-1 contains the indicies of all elements above the pivot
     * Elements n,...,N+1 contain the indicies of all elements below or equal to the pivot (in
     * revrse)
     *
     */
    merian::BufferHandle partition;
    static constexpr vk::BufferUsageFlagBits PARTITION_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    using PartitionLayout =
        layout::StructLayout<storageQualifier,
                             layout::Attribute<wrs::glsl::uint, "heavyCount">,
                             layout::Attribute<wrs::glsl::uint*, "heavyLightIndices">>;
    using PartitionView = layout::BufferView<PartitionLayout>;

    static constexpr vk::DeviceSize minPartitionIndices(std::size_t N) {
        return sizeof(wrs::glsl::uint) * N + sizeof(wrs::glsl::uint);
    }

    static std::size_t partitionSize(std::size_t workgroupSize, std::size_t rows) {
        return workgroupSize * rows;
    }
    static std::size_t workgroupCount(std::size_t elementCount, std::size_t partitionSize) {
        return (elementCount + partitionSize - 1) / partitionSize;
    }

    static DecoupledPrefixPartitionBuffers allocate(merian::ResourceAllocatorHandle alloc,
                                                    std::size_t elementCount,
                                                    std::size_t partitionSize,
                                                    merian::MemoryMappingType memoryMapping);
};

class DecoupledPrefixPartitionConfig {
  public:
    glsl::uint workgroupSize;
    glsl::uint rows;
    glsl::uint parallelLookbackDepth;

    constexpr DecoupledPrefixPartitionConfig()
        : workgroupSize(512), rows(8), parallelLookbackDepth(32) {}
    explicit constexpr DecoupledPrefixPartitionConfig(glsl::uint workgroupSize,
                                                      glsl::uint rows,
                                                      glsl::uint parallelLookbackDepth)
        : workgroupSize(workgroupSize), rows(rows), parallelLookbackDepth(parallelLookbackDepth) {}

    constexpr glsl::uint partitionSize() const {
        return workgroupSize * rows;
    }
};

class DecoupledPrefixPartition {
  public:
    using weight_t = float;
    using Buffers = DecoupledPrefixPartitionBuffers;

    explicit DecoupledPrefixPartition(const merian::ContextHandle& context,
                                      const merian::ShaderCompilerHandle& shaderCompiler,
                                      DecoupledPrefixPartitionConfig config = {})
        : m_partitionSize(config.partitionSize()) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        std::string shaderPath = "src/wrs/algorithm/prefix_partition/decoupled/shader.comp";

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        assert(context->physical_device.physical_device_subgroup_properties.subgroupSize >=
               config.parallelLookbackDepth);
        specInfoBuilder.add_entry(
            config.workgroupSize,
            context->physical_device.physical_device_subgroup_properties.subgroupSize, config.rows,
            config.parallelLookbackDepth);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }
    void run(const merian::CommandBufferHandle& cmd,
             const DecoupledPrefixPartitionBuffers& buffers,
             uint32_t N) {
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.elements, buffers.pivot,
                                 buffers.batchDescriptors, buffers.partitionPrefix,
                                 buffers.partition);
        cmd->push_constant(m_pipeline, N);
        uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    vk::DeviceSize minBufferDescriptorSize(uint32_t N) {
        return DecoupledPrefixPartitionBuffers::minBatchDescriptorSize(N, m_partitionSize,
                                                                       sizeof(weight_t));
    }

    inline glsl::uint getPartitionSize() const {
        return m_partitionSize;
    }

  private:
    const uint32_t m_partitionSize;
    merian::PipelineHandle m_pipeline;
};

} // namespace wrs
