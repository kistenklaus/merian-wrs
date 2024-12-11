#pragma once

#include "src/wrs/generic_types.hpp"
#include "src/wrs/reference/pack.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include <fmt/base.h>
#include <memory>
#include <span>
#include <spdlog/spdlog.h>
#include <vector>
namespace wrs::reference {

template <wrs::arithmetic T,
          std::floating_point P,
          std::integral I,
          wrs::generic_allocator Allocator = std::allocator<void>>
std::vector<wrs::alias_table_entry_t<P, I>,
            typename std::allocator_traits<Allocator>::template rebind_alloc<
                wrs::alias_table_entry_t<P, I>>>
psa_alias_table(const std::span<T> weights_,
                const I K,
                const Allocator& alloc = {}) {

    std::vector<T> weights {weights_.begin(), weights_.end()};
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] *= 0.0001;
    }

    const auto totalWeight = wrs::reference::kahan_reduction<T>(weights);

    using Entry = wrs::alias_table_entry_t<P, I>;
    using EntryAllocator = std::allocator_traits<Allocator>::template rebind_alloc<Entry>;
    using IAllocator = std::allocator_traits<Allocator>::template rebind_alloc<I>;
    using TAllocator = std::allocator_traits<Allocator>::template rebind_alloc<T>;
    using SplitAllocator =
        std::allocator_traits<Allocator>::template rebind_alloc<wrs::split_t<T, I>>;

    const I N = static_cast<I>(weights.size());
    const P averageWeight = static_cast<P>(totalWeight) / static_cast<P>(N);

    auto [heavyIndices, lightIndices, _indices] =
        wrs::reference::stable_partition_indicies<T, I, IAllocator>(weights, averageWeight,
                                                                    IAllocator(alloc));
    auto [heavy, light, _partition] =
        wrs::reference::stable_partition<T, TAllocator>(weights, averageWeight, TAllocator(alloc));

    auto heavyPrefix =
        wrs::reference::prefix_sum<T, TAllocator>(std::span<T>(heavy), true, TAllocator(alloc));
    auto lightPrefix =
        wrs::reference::prefix_sum<T, TAllocator>(std::span<T>(light), true, TAllocator(alloc));

    /* fmt::println("weights = {}", weights); */
    auto splits = wrs::reference::splitK<T, I, SplitAllocator>(
        std::span<T>(heavyPrefix), std::span<T>(lightPrefix), static_cast<T>(averageWeight), N, K,
        SplitAllocator{alloc});

    /* for (size_t k = 0; k < splits.size(); k++) { */
    /*   fmt::println("split[{}] = {}",k, splits[k]); */
    /* } */
    /* assert(false); */

    wrs::alias_table_t<P, I, EntryAllocator> aliasTable =
        wrs::reference::packSplits<T, P, I, EntryAllocator>(
            heavyIndices, lightIndices, weights, averageWeight, splits, EntryAllocator{alloc});

    return aliasTable;
}

namespace pmr {
template <wrs::arithmetic T, std::floating_point P, std::integral I>
std::vector<wrs::alias_table_entry_t<P, I>,
            std::pmr::polymorphic_allocator<wrs::alias_table_entry_t<P, I>>>
psa_alias_table(const std::span<T>& weights,
                const I K,
                const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::reference::psa_alias_table<T, P, I, std::pmr::polymorphic_allocator<void>>(
        weights, K, alloc);
}

} // namespace pmr

} // namespace wrs::reference
