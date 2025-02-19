#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/host/layout/ArrayLayout.hpp"
#include "src/host/layout/BufferView.hpp"
#include "src/host/types/glsl.hpp"
#include <fmt/base.h>

namespace device {

template <typename T> struct BlockCombineBuffers {
    using Self = BlockCombineBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle blockScan;
    using ReductionsLayout = host::layout::ArrayLayout<T, storageQualifier>;
    using ReductionsView = host::layout::BufferView<ReductionsLayout>;

    merian::BufferHandle elementScan;
    using PrefixSumLayout = host::layout::ArrayLayout<T, storageQualifier>;
    using PrefixSumView = host::layout::BufferView<PrefixSumLayout>;
};

struct BlockCombineConfig {
    const host::glsl::uint workgroupSize;
    const host::glsl::uint rows;
    const host::glsl::uint sequentialCombineLength;
    const host::glsl::uint blocksPerWorkgroup;

    constexpr BlockCombineConfig(host::glsl::uint workgroupSize,
                                 host::glsl::uint rows,
                                 host::glsl::uint sequentialCombineLength,
                                 host::glsl::uint blocksPerWorkgroup,
                                 std::optional<host::glsl::uint> blockSizeCheck = std::nullopt)
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

    constexpr host::glsl::uint blockSize() const {
        return workgroupSize * rows * sequentialCombineLength;
    }

    constexpr host::glsl::uint tileSize() const {
        return blockSize() * blocksPerWorkgroup;
    }
};

template <typename T = float> class BlockCombine {
    struct PushConstants {
      host::glsl::uint N;
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
            "src/device/prefix_sum/block_wise/combine/shader.comp";

        std::map<std::string, std::string> defines;

        if constexpr (std::is_same_v<host::glsl::f32, T>) {
            defines["USE_FLOAT"];
        } else if constexpr (std::is_same_v<host::glsl::uint, T>) {
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

    void run(const merian::CommandBufferHandle& cmd, const Buffers& buffers, host::glsl::uint N) {
        cmd->bind(m_pipeline);
        cmd->push_descriptor_set(m_pipeline, buffers.blockScan, buffers.elementScan);
        cmd->push_constant<PushConstants>(m_pipeline, PushConstants{.N = N});
        const uint32_t workgroupCount = (N + m_tileSize - 1) / m_tileSize;
        cmd->dispatch(workgroupCount, 1, 1);
    }

    inline host::glsl::uint blockSize() const {
        return m_tileSize;
    }

  private:
    merian::PipelineHandle m_pipeline;
    host::glsl::uint m_tileSize;
};

} // namespace device
