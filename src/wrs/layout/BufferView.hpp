#pragma once

#include <utility>
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "src/wrs/layout/ArrayLayout.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/layout/layout_traits.hpp"
#include "src/wrs/layout/Attribute.hpp"

namespace wrs::layout {

namespace view_state {
struct BufferViewBarrierState {
  bool postHostWrite = false;
  bool postTransferWrite = false;
  bool postShaderWrite = false;

  BufferViewBarrierState() = default;
};
static_assert(std::semiregular<BufferViewBarrierState>);
using BufferViewBarrierStateHandle = std::shared_ptr<BufferViewBarrierState>;
};

    using namespace wrs::layout::traits;
template <wrs::layout::traits::any_layout Layout>
class BufferView {
  private:

  public:
    BufferView(merian::BufferHandle buffer, Layout layout = {}) requires (traits::IsSizedLayout<Layout>):
        layout(layout), m_buffer(std::move(std::move(buffer))),
        m_arraySize(0), m_barrierState(std::make_shared<view_state::BufferViewBarrierState>()){}
    BufferView(merian::BufferHandle buffer, const std::size_t arraySize, Layout layout = {}) requires (traits::IsUnsizedLayout<Layout>):
        layout(layout), m_buffer(std::move(buffer)),
      m_arraySize(arraySize), m_barrierState(std::make_shared<view_state::BufferViewBarrierState>()){}
    BufferView(merian::BufferHandle buffer, view_state::BufferViewBarrierStateHandle barrierState,
        Layout layout = {}) requires (traits::IsSizedLayout<Layout>): 
        layout(layout), m_buffer(std::move(buffer)),
  m_arraySize(0), m_barrierState(std::move(barrierState)) {}
    BufferView(merian::BufferHandle buffer, std::size_t arraySize, view_state::BufferViewBarrierStateHandle barrierState, Layout layout = {}) requires (traits::IsUnsizedLayout<Layout>): 
      layout(layout), m_buffer(std::move(buffer)),
      m_arraySize(arraySize), m_barrierState(std::move(barrierState)) {}
    
    ~BufferView() = default;
    BufferView(const BufferView&) = default;
    BufferView(BufferView&&) noexcept = default;
    BufferView& operator=(const BufferView&) = default;
    BufferView& operator=(BufferView&&) noexcept = default;

    template <wrs::glsl::primitive_like T>
    void upload(T primitive)
        requires(traits::IsPrimitiveLayout<Layout> && std::same_as<T, typename Layout::base_type>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.setMapped(mapped, primitive);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template<typename T>
    void upload(std::span<const T> primitives)
        requires(traits::IsPrimitiveArrayLayout<Layout> && 
            std::same_as<T, typename Layout::base_type>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.setMapped(mapped, primitives);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template <wrs::layout::traits::IsStorageCompatibleStruct<typename Layout::base_type> S>
    void upload(std::span<const S> structures)
        requires(traits::IsComplexArrayLayout<Layout>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.template setMapped<S>(mapped, structures);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template <wrs::layout::traits::IsStorageCompatibleStruct<Layout> S>
    void upload(const S& s)
        requires(traits::IsSizedStructLayout<Layout>)
    {
        void* mapped = m_buffer->get_memory()->map();
        layout.setMapped(mapped, s);
        m_buffer->get_memory()->unmap();
        m_barrierState->postHostWrite = true;
    }

    template<typename T>
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


    template<typename T, wrs::typed_allocator<T> Allocator = std::allocator<T>>
    std::vector<T, Allocator> 
    download(const Allocator& alloc = {}) requires (
        traits::IsPrimitiveArrayLayout<Layout> &&
          std::same_as<T, typename Layout::base_type>
        ) {
      assert(!m_barrierState->postTransferWrite);
      assert(!m_barrierState->postShaderWrite);
      void* mapped = m_buffer->get_memory()->map();
      std::vector<typename Layout::base_type, Allocator> out = layout.template getFromMapped<Allocator>(mapped, m_arraySize, alloc);
      m_buffer->get_memory()->unmap();
      return out;
    }

    template<wrs::layout::traits::IsStorageCompatibleStruct<typename Layout::base_type> S,
        wrs::typed_allocator<S> Allocator = std::allocator<S>>
    std::vector<S, Allocator> download(const Allocator& alloc = {}) requires (traits::IsComplexArrayLayout<Layout>) {
      assert(!m_barrierState->postTransferWrite);
      assert(!m_barrierState->postShaderWrite);
      void* mapped = m_buffer->get_memory()->map();
      std::vector<S, Allocator> out = layout.template getFromMapped<S, Allocator>(mapped, m_arraySize, alloc);
      m_buffer->get_memory()->unmap();
      return out;
    }


    template<wrs::layout::traits::IsStorageCompatibleStruct<Layout> S>
    S download() requires (traits::IsSizedStructLayout<Layout>) {
      assert(!m_barrierState->postTransferWrite);
      assert(!m_barrierState->postShaderWrite);
      void* mapped = m_buffer->get_memory()->map();
      S out = layout.template getFromMapped<S>(mapped);
      m_buffer->get_memory()->unmap();
      return out;
    }

    template<StaticString AttributeName>
    auto attribute() requires (traits::IsStructLayout<Layout> && traits::IsSizedLayout<Layout>) {
      using AttributeLayout = struct_attribute_type<Layout, AttributeName>;
      return BufferView<AttributeLayout>(m_buffer, m_barrierState, layout.template get<AttributeName>());
    }

    template<StaticString AttributeName>
    auto attribute() requires (traits::IsStructLayout<Layout> && traits::IsUnsizedLayout<Layout>) {
      using AttributeLayout = struct_attribute_type<Layout, AttributeName>;
      if constexpr (traits::IsSizedLayout<AttributeLayout>) {
        return BufferView<AttributeLayout>(m_buffer, m_barrierState, layout.template get<AttributeName>());
      }else {
        return BufferView<AttributeLayout>(m_buffer, m_arraySize, m_barrierState, layout.template get<AttributeName>());
      }
    }

    [[nodiscard]] std::size_t size() const requires (traits::IsSizedLayout<Layout>) {
      return Layout::size();
    }

    [[nodiscard]] std::size_t size() const requires (traits::IsUnsizedLayout<Layout>) {
      return Layout::size(m_arraySize);
    }

    void zero() {
      std::byte* mapped = m_buffer->get_memory()->map_as<std::byte>() + layout.offset( + layout.offset());
      std::memset(mapped, 0, size());
      m_buffer->get_memory()->unmap();
      m_barrierState->postHostWrite = true;
    }

    void zero(vk::CommandBuffer cmd) {
      if (m_barrierState->postShaderWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
            {});
        m_barrierState->postShaderWrite = false;
      }
      if (m_barrierState->postHostWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eTransferRead),
            {});
        m_barrierState->postHostWrite = false;
      }
      cmd.fillBuffer(*m_buffer, 0, size(), 0);
      m_barrierState->postTransferWrite = true;
    }

    void copyTo(vk::CommandBuffer cmd, const merian::BufferHandle& o) {
      if (m_barrierState->postShaderWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
            {});
        m_barrierState->postShaderWrite = false;
      }
      if (m_barrierState->postHostWrite) {
          cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                              vk::PipelineStageFlagBits::eTransfer, {}, {},
                              m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                       vk::AccessFlagBits::eTransferRead),
                              {});
          m_barrierState->postHostWrite = false;
      }

      const vk::BufferCopy copy{0,0,size()};
      cmd.copyBuffer(*m_buffer, *o, 1, &copy);
    }

    template <wrs::layout::traits::any_layout OtherLayout>
    void copyTo(const vk::CommandBuffer cmd, BufferView<OtherLayout>& other) {
      if (m_barrierState->postShaderWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
            {});
        m_barrierState->postShaderWrite = false;
      }
      if (m_barrierState->postHostWrite) {
          cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                              vk::PipelineStageFlagBits::eTransfer, {}, {},
                              m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                       vk::AccessFlagBits::eTransferRead),
                              {});
          m_barrierState->postHostWrite = false;
      }

      const vk::BufferCopy copy{0,0,size()};
      cmd.copyBuffer(*m_buffer, *other.get(), 1, &copy);
      other.expectTransferWrite();
    }

    void expectHostRead(const vk::CommandBuffer cmd) const {
      if (m_barrierState->postShaderWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead),
            {});
        m_barrierState->postShaderWrite = false;
      }
      if (m_barrierState->postTransferWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead),
            {});
        m_barrierState->postTransferWrite = false;
      }
    }

    void expectComputeRead(const vk::CommandBuffer cmd) const {
      if (m_barrierState->postTransferWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
            {});
        m_barrierState->postTransferWrite = false;
      }
      if (m_barrierState->postHostWrite) {
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eComputeShader, {}, {},
            m_buffer->buffer_barrier(vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead),
            {});
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

struct Value {
    float x;
    float y;
    static constexpr glsl::StorageQualifier storage_qualifier = glsl::StorageQualifier::std430;
    static constexpr std::size_t size(glsl::StorageQualifier) {
      return sizeof(Value);
    }
    static constexpr std::size_t alignment(glsl::StorageQualifier) {
      return alignof(Value);
    }
};

static void foo() {
    
    merian::BufferHandle handle;

    using XLayout = PrimitiveLayout<float, glsl::StorageQualifier::std430>;
    using Inner =
        StructLayout<glsl::StorageQualifier::std430, Attribute<float, "x">, Attribute<float, "y">>;
    using Layout = ArrayLayout<Inner, glsl::StorageQualifier::std430>;
    using Layout2 = ArrayLayout<int, glsl::StorageQualifier::std430>;
    static_assert(wrs::layout::traits::IsSizedStructLayout<Inner>);
    static_assert(wrs::layout::traits::IsStorageCompatibleStruct<Value, Inner>);
    static_assert(wrs::layout::traits::IsStructLayout<Inner>);

    static_assert(!wrs::layout::traits::IsPrimitiveArrayLayout<Inner>);

    BufferView<Inner> innerBuffer{handle};
    BufferView<Layout> buffer{handle, 10};
    BufferView<Layout2> buffer2{handle, 10};

    BufferView<XLayout> x = innerBuffer.attribute<"x">();
    x.upload<float>(1.0f);
    /**/
    /*Value v = innerBuffer.download<Value>();*/
    /**/
    /*std::pmr::vector<int> x = buffer2.download<wrs::pmr_alloc<int>>(2);*/
    /**/
    /*std::pmr::vector<Value> v = buffer.download<Value, wrs::pmr_alloc<Value>>(2);*/
    
}

} // namespace wrs::layout
