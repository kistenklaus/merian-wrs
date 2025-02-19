#include "./DefaultResource.hpp"
#include <cstdlib>
#include <memory_resource>
#include <spdlog/spdlog.h>

class DefaultResource : public wrs::memory::MemoryResource {
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        return std::pmr::get_default_resource()->allocate(bytes, alignment);
    }
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        std::pmr::get_default_resource()->deallocate(p, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override {
        return std::pmr::get_default_resource()->is_equal(o);
    }

    wrs::memory::MemoryResource::OwnType do_owns(const void* pointer [[maybe_unused]]) override {
        return wrs::memory::MemoryResource::OWNS_WEAKLY;
    }
};

static DefaultResource defaultDefault{}; // singleton

wrs::memory::MemoryResource* wrs::memory::getDefaultResource() {
    return &defaultDefault;
}
