#pragma once

#include "src/wrs/memory/DefaultResource.hpp"
#include "src/wrs/memory/MemoryResource.hpp"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <unistd.h>
namespace wrs::memory {

class StackResource : public wrs::memory::MemoryResource {

  public:
    StackResource(std::size_t capacity = 4096,
                  wrs::memory::MemoryResource* upstream = wrs::memory::getDefaultResource())
        : m_upstream(upstream), m_space(capacity), m_capacity(capacity) {
        assert(m_upstream != nullptr);
        std::pmr::polymorphic_allocator<std::byte> alloc{upstream};
        m_stack = alloc.allocate(m_capacity);
        m_head = m_stack;
    }

    StackResource(const StackResource&) = delete;
    StackResource(StackResource& o)
        : m_upstream(o.m_upstream), m_head(o.m_head), m_space(o.m_space), m_capacity(o.m_capacity),
          m_stack(o.m_stack) {
        o.m_upstream = nullptr;
        o.m_head = nullptr;
        o.m_space = 0;
        o.m_capacity = 0;
        o.m_stack = nullptr;
    }

    StackResource operator=(const StackResource&) = delete;
    StackResource operator=(StackResource& o) {
        if (this == &o) {
            return *this;
        }
        if (m_stack != nullptr && m_upstream != nullptr) {
            std::pmr::polymorphic_allocator<std::byte> alloc{m_upstream};
            alloc.deallocate(m_stack, m_capacity);
        }
        m_upstream = o.m_upstream;
        m_head = o.m_head;
        m_space = o.m_space;
        m_capacity = o.m_capacity;
        m_stack = o.m_stack;
        o.m_upstream = nullptr;
        o.m_head = nullptr;
        o.m_space = 0;
        o.m_capacity = 0;
        o.m_stack = nullptr;
        return *this;
    }

    ~StackResource() {
        if (m_stack != nullptr && m_upstream != nullptr) {
            std::pmr::polymorphic_allocator<std::byte> alloc{m_upstream};
            alloc.deallocate(m_stack, m_capacity);
        }
    }

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        void* p = m_head;
        if (std::align(alignment, bytes, p, m_space) != nullptr) {
            m_head = reinterpret_cast<std::byte*>(p);
            m_head += bytes;
            m_space -= bytes;
            return p;
        }
        return nullptr;
    }

    void do_deallocate(void* p [[maybe_unused]],
                       std::size_t bytes [[maybe_unused]],
                       std::size_t alignment [[maybe_unused]]) override {}

    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override {
        if (const StackResource* r = dynamic_cast<const StackResource*>(&o); r != nullptr) {
            return this->m_stack == r->m_stack;
        }
        return false;
    }

    OwnType do_owns(const void* pointer) override {
        const std::byte* p = reinterpret_cast<const std::byte*>(pointer);
        return (p >= m_stack && p < (m_stack + m_capacity)) ? OWNS_STRONGLY : DOES_NOT_OWN;
    }

    /**
     * Resets the stack allocator.
     * Does not deallocate!!!
     */
    void reset() {
        m_head = m_stack;
        m_space = m_capacity;
    }

  private:
    wrs::memory::MemoryResource* m_upstream;
    std::byte* m_head;
    std::size_t m_space;
    std::size_t m_capacity;
    std::byte* m_stack;
};

} // namespace wrs::memory
