#pragma once

#include "src/host/memory/DefaultResource.hpp"
#include "src/host/memory/MemoryResource.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>
namespace host::memory {

class SafeResource : public MemoryResource {
  public:
    SafeResource(MemoryResource* upstream = getDefaultResource()) : m_upstream(upstream) {}

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
    MemoryResource* m_upstream;
};

} // namespace host::memory
