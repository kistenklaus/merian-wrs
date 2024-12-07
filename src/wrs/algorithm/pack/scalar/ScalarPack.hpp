#pragma once

#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include <concepts>
#include <memory>
#include <vulkan/vulkan_handles.hpp>
namespace wrs {

struct ScalarPackBuffers {
    merian::BufferHandle splits;

    merian::BufferHandle mean;

    merian::BufferHandle heavyLightPartition;

    merian::BufferHandle aliasTable;
};

template <typename T> class ScalarPack {
    static_assert(std::same_as<T, float>, "Other weights are currently not supported");

  private:
#ifdef NDEBUG
    static constexpr bool CHECK_PARAMETERS = false;
#else
    static constexpr bool CHECK_PARAMETERS = true;
#endif
  public:
    using weight_t = T;

    ScalarPack(const merian::ContextHandle& context) {

        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer()
                .add_binding_storage_buffer()
                .build_push_descriptor_layout(context);

        std::string shaderPath = "src/wrs/algorithm/pack/scalar/float.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<uint32_t>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);

        m_writes.resize(0);
    }

    void run(vk::CommandBuffer cmd, ScalarPackBuffers& buffers) {

        m_pipeline->bind(cmd);
        uint32_t workgroupCount = 0; // TODO
        cmd.dispatch(workgroupCount, 1, 1);
    }

  private:
    merian::PipelineHandle m_pipeline;
    std::vector<vk::WriteDescriptorSet> m_writes;
};

} // namespace wrs
