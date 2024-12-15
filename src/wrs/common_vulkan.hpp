#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
namespace wrs::common_vulkan {

inline void pipelineBarrierTransferReadAfterHostWrite(vk::CommandBuffer cmd,
                                                      merian::BufferHandle buffer) {
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer, {}, {},
        buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eTransferRead),
        {});
}

inline void pipelineBarrierComputeReadAfterTransferWrite(vk::CommandBuffer cmd,
                                                         merian::BufferHandle buffer) {
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {},
        buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
        {});
}

inline void pipelineBarrierTransferReadAfterComputeWrite(vk::CommandBuffer cmd,
                                                         merian::BufferHandle buffer) {
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
        buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
        {});
}

inline void pipelineBarrierHostReadAfterTransferWrite(vk::CommandBuffer cmd,
                                                      merian::BufferHandle buffer) {
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {},
        buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead),
        {});
}

} // namespace wrs::common_vulkan
