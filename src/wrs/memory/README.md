This directory contains some super basic allocator patterns,
it most likely useless, i just wanted to make sure that we don't leak any 
when we run all test suits, therefor we implemented a simple extension 
to the std::pmr::memory_resources with polymorphic allocators,
which primarily contains a StackResource, which is similar 
to a std::pmr::monotonic_buffer_resource, but allows freeing 
the all allocates resources at once. 

Because we use a StackAllocator for most test suits we can guarantee that
no memory leaks the downside is that allocations patterns,
like from std::vector<T>::push_back should be avoided,
which makes the testing suit uselessly complicated.
