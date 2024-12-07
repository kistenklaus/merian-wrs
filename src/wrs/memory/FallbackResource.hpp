#pragma once

#include "src/wrs/memory/DefaultResource.hpp"
#include "src/wrs/memory/MemoryResource.hpp"
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <stdexcept>
namespace wrs::memory {

class FallbackResource : public wrs::memory::MemoryResource {
  public:
    FallbackResource(wrs::memory::MemoryResource* primary,
                     wrs::memory::MemoryResource* fallback = getDefaultResource())
        : m_primaryUpstream(primary), m_fallbackUpstream(fallback) {}

  protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        void* p = m_primaryUpstream->allocate(bytes, alignment);
        if (p != nullptr) {
            return p;
        }
        return m_fallbackUpstream->allocate(bytes, alignment);
    }

    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override {
        OwnType p = m_primaryUpstream->owns(pointer);
        if (p == OWNS_STRONGLY) {
            m_primaryUpstream->deallocate(pointer, bytes, alignment);
            return;
        }
        OwnType f = m_fallbackUpstream->owns(pointer);
        if (f == OWNS_STRONGLY) {
            m_fallbackUpstream->deallocate(pointer, bytes);
            return;
        }
        if (p == OWNS_WEAKLY) {
            m_primaryUpstream->deallocate(pointer, bytes, alignment);
            return;
        }
        if (f == OWNS_WEAKLY) {
            m_fallbackUpstream->deallocate(pointer, bytes);
            return;
        }
        throw std::runtime_error("Deallocation failed: trying to deallocate non owned pointer");
    }

    bool do_is_equal(const std::pmr::memory_resource& resource) const noexcept override {
        // The spec states that:
        // Two memory_resources compare equal if and only if memory allocated from one
        // memory_resource can be deallocated from the other and vice versa.
        //
        // Any resource allocated from a FallbackResource can be deallocated from the underlying
        // resource. This only works because we implement the additional owns property of resources!
        return m_primaryUpstream->is_equal(resource) || m_fallbackUpstream->is_equal(resource);
    }

    OwnType do_owns(const void* pointer) override {
        OwnType p = m_primaryUpstream->owns(pointer);
        OwnType f = m_fallbackUpstream->owns(pointer);
        if (p == OWNS_STRONGLY || f == OWNS_STRONGLY) {
            return OWNS_STRONGLY;
        }
        if (p == OWNS_WEAKLY || f == OWNS_WEAKLY) {
            return OWNS_WEAKLY;
        }
        return DOES_NOT_OWN;
    }

  private:
    wrs::memory::MemoryResource* m_primaryUpstream;
    wrs::memory::MemoryResource* m_fallbackUpstream;
};

} // namespace wrs::memory
