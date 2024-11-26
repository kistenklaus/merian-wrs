#pragma once

#include "src/wrs/memory/DefaultResource.hpp"
#include "src/wrs/memory/MemoryResource.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>
namespace wrs::memory {

class SafeResource : public wrs::memory::MemoryResource {
  public:
    SafeResource(wrs::memory::MemoryResource* upstream = wrs::memory::getDefaultResource())
        : m_upstream(upstream) {}

  protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        void* p = m_upstream->allocate(bytes, alignment);
        if (p == nullptr) {
            throw std::runtime_error("safe allocation failed");
        }
        return p;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        m_upstream->deallocate(p, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override {
        return m_upstream->is_equal(o);
    }

    OwnType do_owns(const void* pointer) override {
        return m_upstream->owns(pointer);
    }

  private:
    wrs::memory::MemoryResource* m_upstream;
};

} // namespace wrs::memory
