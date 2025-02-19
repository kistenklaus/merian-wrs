#pragma once

#include "src/host/reference/pack.hpp"
#include "src/host/reference/partition.hpp"
#include "src/host/reference/prefix_sum.hpp"
#include "src/host/reference/reduce.hpp"
#include "src/host/reference/split.hpp"
#include <fmt/base.h>
#include <memory>
#include <span>
#include <spdlog/spdlog.h>
#include <vector>

namespace host::reference {

template <arithmetic T,
          std::floating_point P,
          std::integral I,
          generic_allocator Allocator = std::allocator<void>>
AliasTable<P,
           I,
           typename std::allocator_traits<Allocator>::template rebind_alloc<AliasTableEntry<P, I>>>
psa_alias_table(std::span<const T> weights_, const I K, const Allocator& alloc = {}) {

    std::vector<T> weights{weights_.begin(), weights_.end()};
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] *= 0.0001;
    }

    const auto totalWeight = reference::kahan_reduction<T>(weights);

    using Entry = AliasTableEntry<P, I>;
    using EntryAllocator = std::allocator_traits<Allocator>::template rebind_alloc<Entry>;
    using IAllocator = std::allocator_traits<Allocator>::template rebind_alloc<I>;
    using TAllocator = std::allocator_traits<Allocator>::template rebind_alloc<T>;
    using SplitAllocator = std::allocator_traits<Allocator>::template rebind_alloc<Split<T, I>>;

    const I N = static_cast<I>(weights.size());
    const P averageWeight = static_cast<P>(totalWeight) / static_cast<P>(N);

    const auto heavyLightIndices = reference::stable_partition_indicies<T, I, IAllocator>(
        weights, averageWeight, IAllocator(alloc));
    const auto heavyLightPartition =
        reference::stable_partition<T, TAllocator>(weights, averageWeight, TAllocator(alloc));

    const std::vector<T, TAllocator> heavyPrefix =
        reference::prefix_sum<T, TAllocator>(heavyLightPartition.heavy(), TAllocator(alloc));
    const std::vector<T, TAllocator> lightPrefix =
        reference::prefix_sum<T, TAllocator>(heavyLightPartition.light(), TAllocator(alloc));

    /* fmt::println("weights = {}", weights); */
    const std::vector<Split<T, I>, SplitAllocator> splits = reference::splitK<T, I, SplitAllocator>(
        heavyPrefix, lightPrefix, static_cast<T>(averageWeight), N, K, SplitAllocator{alloc});

    /* for (size_t k = 0; k < splits.size(); k++) { */
    /*   fmt::println("split[{}] = ({},{},{})",k, splits[k].i, splits[k].j, splits[k].spill); */
    /* } */
    /* fmt::println("lightCount = {}, heavyCount = {}", lightPrefix.size(), heavyPrefix.size()); */
    /* assert(false); */

    const AliasTable<P, I, EntryAllocator> aliasTable =
        reference::packSplits<T, P, I, EntryAllocator>(
            heavyLightIndices.heavy(), heavyLightIndices.light(), weights, averageWeight, splits,
            EntryAllocator{alloc});

    return aliasTable;
}

namespace pmr {
template <arithmetic T, std::floating_point P, std::integral I>
host::pmr::AliasTable<P, I>
psa_alias_table(std::span<const T> weights,
                const I K,
                const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    assert(K < weights.size());
    return reference::psa_alias_table<T, P, I, std::pmr::polymorphic_allocator<void>>(weights, K,
                                                                                      alloc);
}

} // namespace pmr

} // namespace host::reference
