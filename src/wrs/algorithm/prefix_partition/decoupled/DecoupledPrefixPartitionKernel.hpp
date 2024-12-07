#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include <stdexcept>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>

namespace wrs {

struct DecoupledPrefixPartitionBuffers {
    /**
     * Buffer, which contains all elements
     * over which the partition and the
     * prefix sums are computed.
     * The buffer is requires to have a packed layout!.
     *
     * Layout : T[[N]]
     *
     * MinSize: sizeof(T) * N
     */
    merian::BufferHandle elements;
    static constexpr vk::BufferUsageFlagBits ELEMENT_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;

    /**
     * Buffer, which contains the pivot element.
     *
     * Layout :  T[[0]] = pivot
     *
     * MinSize: sizeof(T)
     */
    merian::BufferHandle pivot;
    static constexpr vk::BufferUsageFlagBits PIVOT_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;

    /**
     * Buffer, which holds internal state of the decoupled lookback.
     * Before a kernel is executed it is required that this buffer is zeroed!
     *
     * Layout : implementation defined.
     * MinSize: ((DecoupledPrefixPartitionKernel)instance).minBufferDescriptorSize(N)
     */
    merian::BufferHandle batchDescriptors;
    static constexpr vk::BufferUsageFlagBits BATCH_DESCRIPTOR_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
    constexpr static vk::DeviceSize
    minBatchDescriptorSize(uint32_t N, uint32_t partitionSize, size_t sizeof_weight) {
        uint32_t workgroupCount = (N + partitionSize - 1) / partitionSize;
        vk::DeviceSize batchDescriptorSize = 4 * sizeof_weight + 2 * sizeof_weight + sizeof_weight;
        // TODO consider proper padding this is just a upper bound for a guess!
        batchDescriptorSize += sizeof(uint32_t) * 4;
        return batchDescriptorSize * workgroupCount;
    }
    constexpr static vk::DeviceSize minBatchDescriptorSize(uint32_t N,
                                                           uint32_t workgroupSize,
                                                           uint32_t rows,
                                                           size_t sizeof_weight) {
        const uint32_t partitionSize = workgroupSize * rows;
        return minBatchDescriptorSize(N, partitionSize, sizeof_weight);
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

    /**
     * Buffer, which contains both partitions.
     * Must be set iff. WRITE_PARTITION is set.
     *
     * Layout: T[[N]]
     * Elements 0,...n-1 contains the elements over all elements above the pivot
     * Elements n,...,N+1 contain the elements over all elements
     *
     */
    std::optional<merian::BufferHandle> partition;
    static constexpr vk::BufferUsageFlagBits PARTITION_BUFFER_USAGE_FLAGS =
        vk::BufferUsageFlagBits::eStorageBuffer;
};

template <typename T = float> class DecoupledPrefixPartition {
    /* static_assert(std::is_same<T, float>(), "Currently only floats as weights are supported"); */

#ifdef NDEBUG
    static constexpr bool CHECK_PARAMETERS = false;
#else
    static constexpr bool CHECK_PARAMETERS = true;
#endif

  public:
    using weight_t = T;
    static constexpr uint32_t DEFAULT_WORKGROUP_SIZE = 512;
    static constexpr uint32_t DEFAULT_ROWS = 4;

    DecoupledPrefixPartition(const merian::ContextHandle& context,
                             uint32_t workgroupSize = DEFAULT_WORKGROUP_SIZE,
                             uint32_t rows = DEFAULT_ROWS,
                             bool writePartition = false,
                             bool stable = false)
        : m_partitionSize(workgroupSize * rows), m_writePartition(writePartition), m_stable(stable),
          m_writes() {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        std::string shaderPath;
        if (writePartition) {
            if (m_stable) {
                shaderPath = "src/wrs/algorithm/prefix_partition/decoupled/"
                             "float_stable_write_partition.comp";
            } else {
                shaderPath = "src/wrs/algorithm/prefix_partition/decoupled/"
                             "float_unstable_write_partition.comp";
            }
        } else {
            if (m_stable) {
                shaderPath = "src/wrs/algorithm/prefix_partition/decoupled/float_stable.comp";
            } else {
                shaderPath = "src/wrs/algorithm/prefix_partition/decoupled/float_unstable.comp";
            }
        }

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(
            workgroupSize,
            context->physical_device.physical_device_subgroup_properties.subgroupSize, rows);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
        m_writes.resize(m_writePartition ? 5 : 4);
        vk::WriteDescriptorSet& elementsDescriptor = m_writes[0];
        elementsDescriptor.setDstBinding(0);
        elementsDescriptor.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& pivotDescriptor = m_writes[1];
        pivotDescriptor.setDstBinding(1);
        pivotDescriptor.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& batchStateDescriptor = m_writes[2];
        batchStateDescriptor.setDstBinding(2);
        batchStateDescriptor.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        vk::WriteDescriptorSet& prefixPartitionDescriptor = m_writes[3];
        prefixPartitionDescriptor.setDstBinding(3);
        prefixPartitionDescriptor.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        if (m_writePartition) {
            vk::WriteDescriptorSet& partitionDescriptor = m_writes[4];
            partitionDescriptor.setDstBinding(4);
            partitionDescriptor.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        }
    }
    void run(vk::CommandBuffer cmd, const DecoupledPrefixPartitionBuffers& buffers, uint32_t N) {
        if constexpr (CHECK_PARAMETERS) {
            // CHECK for VK_NULL_HANDLE
            if (cmd == VK_NULL_HANDLE) {
                throw std::runtime_error("cmd is VK_NULL_HANDLE");
            }
            if (buffers.elements->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.elements is VK_NULL_HANDLE");
            }
            if (buffers.pivot->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.pivot is VK_NULL_HANDLE");
            }
            if (buffers.batchDescriptors->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.batchDescriptors is VK_NULL_HANDLE");
            }
            if (buffers.partitionPrefix->get_buffer() == VK_NULL_HANDLE) {
                throw std::runtime_error("buffers.partitionPrefix is VK_NULL_HANDLE");
            }
            if (m_writePartition) {
                if (!buffers.partition.has_value()) {
                    throw std::runtime_error("buffers.partition is std::nullopt");
                }
                if (buffers.partition.value()->get_buffer() == VK_NULL_HANDLE) {
                    throw std::runtime_error("buffers.partition is VK_NULL_HANDLE");
                }
            }
            // CHECK buffer sizes
            if (buffers.elements->get_size() < sizeof(weight_t) * N) {
                throw std::runtime_error("buffers.elements is to small!");
            }
            if (buffers.pivot->get_size() < sizeof(weight_t)) {
                throw std::runtime_error("buffers.pivot is to small!");
            }
            if (buffers.batchDescriptors->get_size() < minBufferDescriptorSize(N)) {
                throw std::runtime_error("buffers.batchDescriptors is to small!");
            }
            if (buffers.partitionPrefix->get_size() < sizeof(weight_t) + sizeof(weight_t) * N) {
                throw std::runtime_error("buffers.partitionPrefix is to small");
            }
            if (m_writePartition) {
                if (buffers.partition.value()->get_size() < sizeof(weight_t) * N) {
                    throw std::runtime_error("buffers.partition is to small");
                }
            }
        }
        m_pipeline->bind(cmd);
        vk::DescriptorBufferInfo elementDesc = buffers.elements->get_descriptor_info();
        m_writes[0].setBufferInfo(elementDesc);
        vk::DescriptorBufferInfo pivotDesc = buffers.pivot->get_descriptor_info();
        m_writes[1].setBufferInfo(pivotDesc);
        vk::DescriptorBufferInfo batchDesc = buffers.batchDescriptors->get_descriptor_info();
        m_writes[2].setBufferInfo(batchDesc);
        vk::DescriptorBufferInfo prefixDesc = buffers.partitionPrefix->get_descriptor_info();
        m_writes[3].setBufferInfo(prefixDesc);
        vk::DescriptorBufferInfo partitionDesc;
        if (m_writePartition) {
            partitionDesc = buffers.partition.value()->get_descriptor_info();
            m_writes[4].setBufferInfo(partitionDesc);
        }
        m_pipeline->push_descriptor_set(cmd, m_writes);
        m_pipeline->push_constant(cmd, N);

        uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

    vk::DeviceSize minBufferDescriptorSize(uint32_t N) {
        return DecoupledPrefixPartitionBuffers::minBatchDescriptorSize(N, m_partitionSize,
                                                                       sizeof(weight_t));
    }

  private:
    const uint32_t m_partitionSize;
    const bool m_writePartition;
    const bool m_stable; // TODO does not need to be stored!! only required during construction
    merian::PipelineHandle m_pipeline;
    std::vector<vk::WriteDescriptorSet> m_writes;
};

/* namespace pmr { */

/* template <typename T> */
/* using DecoupledPrefixPartitionKernel = */
/*     wrs::DecoupledPrefixPartitionKernel<T,
 * std::pmr::polymorphic_allocator<vk::WriteDescriptorSet>>; */
/* } */

} // namespace wrs
