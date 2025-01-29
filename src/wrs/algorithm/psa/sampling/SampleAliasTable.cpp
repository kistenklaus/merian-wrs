#include "./SampleAliasTable.hpp"

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

wrs::SampleAliasTableBuffers wrs::SampleAliasTableBuffers::allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t weightCount,
                         std::size_t sampleCount) {
        using Self = wrs::SampleAliasTableBuffers;
        Self buffers;
        if (memoryMapping == merian::MemoryMappingType::NONE) {
           buffers.aliasTable = alloc->createBuffer(Self::AliasTableLayout::size(weightCount),  // bye bye LSP!
               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, 
               merian::MemoryMappingType::NONE); 
          buffers.samples = alloc->createBuffer(Self::SamplesLayout::size(sampleCount),
              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
              merian::MemoryMappingType::NONE);
        } else { 
           buffers.aliasTable = alloc->createBuffer(Self::AliasTableLayout::size(weightCount),  // bye bye LSP!
               vk::BufferUsageFlagBits::eTransferSrc, 
               memoryMapping); 
          buffers.samples = alloc->createBuffer(Self::SamplesLayout::size(sampleCount),
              vk::BufferUsageFlagBits::eTransferDst,
              memoryMapping);
        }
        return buffers;
}


wrs::SampleAliasTable::SampleAliasTable(const merian::ContextHandle& context, glsl::uint workgroupSize)
        : m_workgroupSize(workgroupSize) {
        const merian::DescriptorSetLayoutHandle descriptorSet0Layout =
            merian::DescriptorSetLayoutBuilder()
                .add_binding_storage_buffer() // alias table
                .add_binding_storage_buffer() // samples
                .build_push_descriptor_layout(context);

        const std::string shaderPath = "src/wrs/algorithm/psa/sampling/shader.comp";

        const merian::ShaderModuleHandle shader =
            context->shader_compiler->find_compile_glsl_to_shadermodule(
                context, shaderPath, vk::ShaderStageFlagBits::eCompute);

        const merian::PipelineLayoutHandle pipelineLayout =
            merian::PipelineLayoutBuilder(context)
                .add_descriptor_set_layout(descriptorSet0Layout)
                .add_push_constant<PushConstants>()
                .build_pipeline_layout();

        merian::SpecializationInfoBuilder specInfoBuilder;
        specInfoBuilder.add_entry(m_workgroupSize);
        const merian::SpecializationInfoHandle specInfo = specInfoBuilder.build();

        m_pipeline = std::make_shared<merian::ComputePipeline>(pipelineLayout, shader, specInfo);
}


void wrs::SampleAliasTable::run(const vk::CommandBuffer cmd, const Buffers& buffers,
    std::size_t N, std::size_t S, glsl::uint seed) {

    m_pipeline->bind(cmd);
    m_pipeline->push_descriptor_set(cmd, buffers.aliasTable, buffers.samples);
    m_pipeline->push_constant<PushConstants>(cmd, PushConstants{.N = N, .S = S, .seed = seed});
    const uint32_t workgroupCount = (S + m_workgroupSize - 1) / m_workgroupSize;
    cmd.dispatch(workgroupCount, 1, 1);
}
