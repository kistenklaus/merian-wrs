#pragma once

#include <memory_resource>
namespace wrs::memory {

class MemoryResource : public std::pmr::memory_resource {
  public:
    enum OwnType {
        // This memory resource is the only owning resource of a pointer. It has to
        // be used to deallocate a pointer!
        OWNS_STRONGLY,
        // This memory resource owns the pointer, but there might be other resources,
        // which also own the pointer. Therefor it is possible to deallocate a pointer
        // with this resource!
        OWNS_WEAKLY,
        // This memory resource does not own the pointer. It can't be used to
        // deallocate a pointer!
        DOES_NOT_OWN,
    };
    OwnType owns(const void* p) {
        return do_owns(p);
    }

  protected:
    virtual OwnType do_owns(const void* p) = 0;
};

} // namespace wrs::memory
