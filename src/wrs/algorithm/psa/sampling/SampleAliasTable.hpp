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
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/types/glsl.hpp"
#include <concepts>
#include <memory>
#include <vulkan/vulkan_handles.hpp>

#include "merian/vk/memory/resource_allocator.hpp"

namespace wrs {

struct SampleAliasTableBuffers {
    using Self = SampleAliasTableBuffers;
    static constexpr auto storageQualifier = glsl::StorageQualifier::std430;

    merian::BufferHandle aliasTable;
    using _AliasTableEntryLayout = layout::
        StructLayout<storageQualifier, layout::Attribute<float, layout::StaticString("p")>, layout::Attribute<int, layout::StaticString("a")>>;
    using AliasTableLayout = layout::ArrayLayout<_AliasTableEntryLayout, storageQualifier>;
    using AliasTableView = layout::BufferView<AliasTableLayout>;

    merian::BufferHandle samples;
    using SamplesLayout = layout::ArrayLayout<glsl::uint, storageQualifier>;
    using SamplesView = layout::BufferView<SamplesLayout>;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         std::size_t weightCount,
                         std::size_t sampleCount);
};

class SampleAliasTable {
    struct PushConstants {
        glsl::uint N;
        glsl::uint S;
    };

  public:
    using Buffers = SampleAliasTableBuffers;

    explicit SampleAliasTable(const merian::ContextHandle& context, glsl::uint workgroupSize);

    void run(const vk::CommandBuffer cmd, const Buffers& buffers,
        std::size_t N, std::size_t S);

  private:
    merian::PipelineHandle m_pipeline;
    glsl::uint m_workgroupSize;
};

} // namespace wrs
