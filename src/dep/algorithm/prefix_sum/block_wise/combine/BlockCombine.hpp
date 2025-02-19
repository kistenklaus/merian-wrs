#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include "src/wrs/algorithm/prefix_sum/block_scan/BlockScan.hpp"
#include "src/wrs/types/glsl.hpp"
#include <fmt/base.h>
namespace wrs {

template <typename T> struct BlockCombineBuffers {
    using Self = BlockCombineBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle blockScan;
    using ReductionsLayout = layout::ArrayLayout<T, storageQualifier>;
    using ReductionsView = layout::BufferView<ReductionsLayout>;

    merian::BufferHandle elementScan;
    using PrefixSumLayout = layout::ArrayLayout<T, storageQualifier>;
    using PrefixSumView = layout::BufferView<PrefixSumLayout>;
};

struct BlockCombineConfig {
    glsl::uint workgroupSize;
    glsl::uint rows;
    glsl::uint sequentialCombineLength;
    glsl::uint blocksPerWorkgroup;

    constexpr BlockCombineConfig(glsl::uint workgroupSize,
                                 glsl::uint rows,
                                 glsl::uint sequentialCombineLength,
                                 glsl::uint blocksPerWorkgroup,
                                 std::optional<glsl::uint> blockSizeCheck = std::nullopt)
        : workgroupSize(workgroupSize), rows(rows),
          sequentialCombineLength(sequentialCombineLength), blocksPerWorkgroup(blocksPerWorkgroup) {
        if (blockSizeCheck.has_value()) {
            assert(blockSizeCheck.value() == blockSize());
        }
    }

    constexpr BlockCombineConfig(BlockScanConfig blockScanConfig)
        : workgroupSize(blockScanConfig.workgroupSize),
          rows(blockScanConfig.rows * blockScanConfig.sequentialScanLength),
          sequentialCombineLength(1), blocksPerWorkgroup(1) {}

    inline glsl::uint blockSize() const {
        return workgroupSize * rows * sequentialCombineLength;
    }

    inline glsl::uint tileSize() const {
        return blockSize() * blocksPerWorkgroup;
    }
};

template <typename T = float> class BlockCombine {
    struct PushConstants {
        glsl::uint N;
    };

  public:
    using Buffers = BlockCombineBuffers<T>;

    explicit BlockCombine(const merian::ContextHandle& context,
                          const merian::ShaderCompilerHandle& shaderCompiler,
                          BlockCombineConfig config)
        : m_tileSize(config.tileSize()) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // reductions
                .add_binding_storage_buffer() // prefix sum
                .build_push_descriptor_layout(context);

        const std::string shaderPath =
            "src/wrs/algorithm/prefix_sum/block_wise/combine/shader.comp";

        std::map<std::string, std::string> defines;

        if constexpr (std::is_same_v<glsl::f32, T>) {
            defines["USE_FLOAT"];
        } else if constexpr (std::is_same_v<glsl::uint, T>) {
            defines["USE_UINT"];
        } else {
            throw std::runtime_error("unsupported block combine base type");
        }

        const merian::ShaderModuleHandle shader = shaderCompiler->find_compile_glsl_to_shadermodule(
            context, shaderPath, vk::ShaderStageFlagBits::eCompute, {}, defines);

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
        specInfoBuilder.add_entry(config.blocksPerWorkgroup);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
    }

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, glsl::uint N) {
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.blockScan, buffers.elementScan);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_tileSize - 1) / m_tileSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline glsl::uint blockSize() const {
        return m_tileSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_tileSize;
};

} // namespace wrs
