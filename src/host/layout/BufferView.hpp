#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/memory_allocator.hpp" // pragma: keep
#include "merian/vk/memory/resource_allocations.hpp"
#include "src/host/layout/layout_traits.hpp"
#include "src/host/why.hpp"
#include <utility>

namespace host::layout {

namespace view_state {
struct BufferViewBarrierState {
    bool postHostWrite = false;
    bool postTransferWrite = false;
    bool postShaderWrite = false;

    BufferViewBarrierState() = default;
};
static_assert(std::semiregular<BufferViewBarrierState>);
using BufferViewBarrierStateHandle = std::shared_ptr<BufferViewBarrierState>;
}; // namespace view_state

using namespace layout::traits;
template <layout::traits::any_layout Layout> class BufferView {
  private:
  public:
    BufferView(merian::BufferHandle buffer, Layout layout = {})
        requires(traits::IsSizedLayout<Layout>)
        : layout(layout), m_buffer(std::move(std::move(buffer))), m_arraySize(0),
          m_barrierState(std::make_shared<view_state::BufferViewBarrierState>()) {}
    BufferView(merian::BufferHandle buffer, const std::size_t arraySize, Layout layout = {})
        requires(traits::IsUnsizedLayout<Layout>)
        : layout(layout), m_buffer(std::move(buffer)), m_arraySize(arraySize),
          m_barrierState(std::make_shared<view_state::BufferViewBarrierState>()) {}
    BufferView(merian::BufferHandle buffer,
               view_state::BufferViewBarrierStateHandle barrierState,
               Layout layout = {})
        requires(traits::IsSizedLayout<Layout>)
        : layout(layout), m_buffer(std::move(buffer)), m_arraySize(0),
          m_barrierState(std::move(barrierState)) {}
    BufferView(merian::BufferHandle buffer,
               std::size_t arraySize,
               view_state::BufferViewBarrierStateHandle barrierState,
               Layout layout = {})
        requires(traits::IsUnsizedLayout<Layout>)
        : layout(layout), m_buffer(std::move(buffer)), m_arraySize(arraySize),
          m_barrierState(std::move(barrierState)) {}

    ~BufferView() = default;
    BufferView(const BufferView&) = default;
    BufferView(BufferView&&) noexcept = default;
    BufferView& operator=(const BufferView&) = default;
    BufferView& operator=(BufferView&&) noexcept = default;

    template <glsl::primitive_like T>
    void upload(T primitive)
        requires(traits::IsPrimitiveLayout<Layout> && std::same_as<T, typename Layout::base_type>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.setMapped(mapped, primitive);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template <typename T>
    void upload(std::span<const T> primitives)
        requires(traits::IsPrimitiveArrayLayout<Layout> &&
                 std::same_as<T, typename Layout::base_type>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.setMapped(mapped, primitives);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template <layout::traits::IsStorageCompatibleStruct<typename Layout::base_type> S>
    void upload(std::span<const S> structures)
        requires(traits::IsComplexArrayLayout<Layout>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.template setMapped<S>(mapped, structures);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template <layout::traits::IsStorageCompatibleStruct<Layout> S>
    void upload(const S& s)
        requires(traits::IsSizedStructLayout<Layout>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.setMapped(mapped, s);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template <typename T>
    auto download()
        requires(traits::IsPrimitiveLayout<Layout> && std::same_as<T, typename Layout::base_type>)
    {
        assert(!m_barrierState->postTransferWrite);
        assert(!m_barrierState->postShaderWrite);
        void* mapped = m_buffer->get_memory()->map();
        auto out = layout.getFromMapped(mapped);
        m_buffer->get_memory()->unmap();
        return out;
    }

    template <typename T, typed_allocator<T> Allocator = std::allocator<T>>
    std::vector<T, Allocator> download(const Allocator& alloc = {})
        requires(traits::IsPrimitiveArrayLayout<Layout> &&
                 std::same_as<T, typename Layout::base_type>)
    {
        assert(!m_barrierState->postTransferWrite);
        assert(!m_barrierState->postShaderWrite);
        void* mapped = m_buffer->get_memory()->map();
        std::vector<typename Layout::base_type, Allocator> out =
            layout.template getFromMapped<Allocator>(mapped, m_arraySize, alloc);
        m_buffer->get_memory()->unmap();
        return out;
    }

    template <IsStorageCompatibleStruct<typename Layout::base_type> S,
              typed_allocator<S> Allocator = std::allocator<S>>
    std::vector<S, Allocator> download(const Allocator& alloc = {})
        requires(traits::IsComplexArrayLayout<Layout>)
    {
        assert(!m_barrierState->postTransferWrite);
        assert(!m_barrierState->postShaderWrite);
        void* mapped = m_buffer->get_memory()->map();
        std::vector<S, Allocator> out =
            layout.template getFromMapped<S, Allocator>(mapped, m_arraySize, alloc);
        m_buffer->get_memory()->unmap();
        return out;
    }

    template <layout::traits::IsStorageCompatibleStruct<Layout> S>
    S download()
        requires(traits::IsSizedStructLayout<Layout>)
    {
        assert(!m_barrierState->postTransferWrite);
        assert(!m_barrierState->postShaderWrite);
        void* mapped = m_buffer->get_memory()->map();
        S out = layout.template getFromMapped<S>(mapped);
        m_buffer->get_memory()->unmap();
        return out;
    }

    template <StaticString AttributeName>
    auto attribute()
        requires(traits::IsStructLayout<Layout> && traits::IsSizedLayout<Layout>)
    {
        using AttributeLayout = struct_attribute_type<Layout, AttributeName>;
        return BufferView<AttributeLayout>(m_buffer, m_barrierState,
                                           layout.template get<AttributeName>());
    }

    template <StaticString AttributeName>
    auto attribute()
        requires(traits::IsStructLayout<Layout> && traits::IsUnsizedLayout<Layout>)
    {
        using AttributeLayout = struct_attribute_type<Layout, AttributeName>;
        if constexpr (traits::IsSizedLayout<AttributeLayout>) {
            return BufferView<AttributeLayout>(m_buffer, m_barrierState,
                                               layout.template get<AttributeName>());
        } else {
            return BufferView<AttributeLayout>(m_buffer, m_arraySize, m_barrierState,
                                               layout.template get<AttributeName>());
        }
    }

    [[nodiscard]] std::size_t size() const
        requires(traits::IsSizedLayout<Layout>)
    {
        return Layout::size();
    }

    [[nodiscard]] std::size_t size() const
        requires(traits::IsUnsizedLayout<Layout>)
    {
        return Layout::size(m_arraySize);
    }

    void zero() {
        std::byte* mapped =
            m_buffer->get_memory()->map_as<std::byte>() + layout.offset(+layout.offset());
        std::memset(mapped, 0, size());
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    void zero(const merian::CommandBufferHandle& cmd) {
        if (m_barrierState->postShaderWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eTransfer,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eTransferRead));
            m_barrierState->postShaderWrite = false;
        }
        if (m_barrierState->postHostWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                  vk::AccessFlagBits::eTransferRead));
            m_barrierState->postHostWrite = false;
        }
        cmd->fill(m_buffer, 0);
        m_barrierState->postTransferWrite = true;
    }

    void copyTo(const merian::CommandBufferHandle& cmd, const merian::BufferHandle& o) {
        if (m_barrierState->postShaderWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eTransfer,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eTransferRead));
            m_barrierState->postShaderWrite = false;
        }
        if (m_barrierState->postHostWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                  vk::AccessFlagBits::eTransferRead));
            m_barrierState->postHostWrite = false;
        }

        const vk::BufferCopy copy{0, 0, size()};
        cmd->copy(m_buffer, o, copy);
    }

    template <layout::traits::any_layout OtherLayout>
    void copyTo(const merian::CommandBufferHandle& cmd, BufferView<OtherLayout>& other) {
        if (m_barrierState->postShaderWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eTransfer,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eTransferRead));
            m_barrierState->postShaderWrite = false;
        }
        if (m_barrierState->postHostWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                  vk::AccessFlagBits::eTransferRead));
            m_barrierState->postHostWrite = false;
        }

        const vk::BufferCopy copy{0, 0, size()};
        cmd->copy(m_buffer, other.get(), copy);
        other.expectTransferWrite();
    }

    void expectHostRead(const merian::CommandBufferHandle& cmd) const {
        if (m_barrierState->postShaderWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eHost,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eHostRead));
            m_barrierState->postShaderWrite = false;
        }
        if (m_barrierState->postTransferWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                  vk::AccessFlagBits::eHostRead));
            m_barrierState->postTransferWrite = false;
        }
    }

    void expectComputeRead(const merian::CommandBufferHandle& cmd) const {
        if (m_barrierState->postTransferWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eComputeShader,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                  vk::AccessFlagBits::eShaderRead));
            m_barrierState->postTransferWrite = false;
        }
        if (m_barrierState->postHostWrite) {
            cmd->barrier(vk::PipelineStageFlagBits::eHost,
                         vk::PipelineStageFlagBits::eComputeShader,
                         m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                  vk::AccessFlagBits::eShaderRead));
            m_barrierState->postHostWrite = false;
        }
    }

    void expectComputeWrite() const {
        m_barrierState->postShaderWrite = true;
    }

    void expectTransferWrite() const {
        m_barrierState->postTransferWrite = true;
    }

    void expectHostWrite() const {
        m_barrierState->postHostWrite = true;
    }

    [[nodiscard]] const merian::BufferHandle& get() const {
        return m_buffer;
    }

    const Layout layout;

  private:
    merian::BufferHandle m_buffer;
    std::size_t m_arraySize;
    view_state::BufferViewBarrierStateHandle m_barrierState;
};

} // namespace host::layout
