#pragma once

#include "src/wrs/algorithm/prefix_sum/block_wise/block_scan/BlockScan.hpp"
namespace wrs {

struct BlockCombineBuffers {
    using Self = BlockScanBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle blockScan;
    using ReductionsLayout = layout::ArrayLayout<float, storageQualifier>;
    using ReductionsView = layout::BufferView<ReductionsLayout>;

    merian::BufferHandle elementScan;
    using PrefixSumLayout = layout::ArrayLayout<float, storageQualifier>;
    using PrefixSumView = layout::BufferView<PrefixSumLayout>;
};

struct BlockCombineConfig {
    glsl::uint workgroupSize;
    glsl::uint rows;
    glsl::uint sequentialCombineLength;

    constexpr BlockCombineConfig(glsl::uint workgroupSize,
                                 glsl::uint rows,
                                 glsl::uint sequentialCombineLength,
                                 std::optional<glsl::uint> blockSizeCheck = std::nullopt)
        : workgroupSize(workgroupSize), rows(rows),
          sequentialCombineLength(sequentialCombineLength) {
        if (blockSizeCheck.has_value()) {
            assert(blockSizeCheck.value() == partitionSize());
        }
    }

    constexpr BlockCombineConfig(BlockScanConfig blockScanConfig)
        : workgroupSize(blockScanConfig.workgroupSize),
          rows(blockScanConfig.rows * 2 * blockScanConfig.sequentialScanLength),
          sequentialCombineLength(1) {}

    inline glsl::uint partitionSize() const {
        return workgroupSize * rows * sequentialCombineLength;
    }
};

class BlockCombine {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = BlockCombineBuffers;

    explicit BlockCombine(const merian::ContextHandle& context, BlockCombineConfig config)
        : m_partitionSize(config.partitionSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // reductions
                .add_binding_storage_buffer() // prefix sum
                .build_push_descriptor_layout(context);

        const std::string shaderPath =
            "src/wrs/algorithm/prefix_sum/block_wise/combine/shader.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

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
        specInfoBuilder.add_entry(config.sequentialCombineLength);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const vk::CommandBuffer cmd, const Buffers& buffers, glsl::uint N) {

        m_pipeline->bind(cmd);
        m_pipeline->push_descriptor_set(cmd, buffers.blockScan, buffers.elementScan);
        m_pipeline->push_constant<PushConstants>(cmd, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_partitionSize - 1) / m_partitionSize;
        cmd.dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint blockSize() const {
        return m_partitionSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_partitionSize;
};

} // namespace wrs
