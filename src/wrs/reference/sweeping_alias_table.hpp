#pragma once

#include "src/wrs/types/alias_table.hpp"
#include "src/wrs/why.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <memory>
#include <span>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vector>
namespace wrs::reference {

namespace sweeping_internal {

template <wrs::arithmetic T, std::floating_point P, std::integral I>
static I nextLight(const std::span<T>& weights, I i, P pivot) {
    const I N = static_cast<I>(weights.size());
    if (i >= N) {
        return N;
    }
    assert(i < N);
    const auto it = std::find_if(weights.begin() + i, weights.end(),
                                 [pivot](const auto& w) { return w <= pivot; });
    if (it == weights.end()) {
        return N;
    } else {
        return static_cast<I>(it - weights.begin());
    }
}
template <wrs::arithmetic T, std::floating_point P, std::integral I>
static I nextHeavy(const std::span<T>& weights, I i, P pivot) {
    const I N = static_cast<I>(weights.size());
    if (i >= N) {
        return N;
    }
    assert(i < N);
    const auto it = std::find_if(weights.begin() + i, weights.end(),
                                 [pivot](const auto& w) { return w > pivot; });
    if (it == weights.end()) {
        return N;
    } else {
        return static_cast<I>(it - weights.begin());
    }
}

} // namespace sweeping_internal

template <wrs::arithmetic T,
          std::floating_point P,
          std::integral I,
          wrs::generic_allocator Allocator = std::allocator<void>>
wrs::AliasTable<
    P,
    I,
    typename std::allocator_traits<Allocator>::template rebind_alloc<wrs::AliasTableEntry<P, I>>>
sweeping_alias_table(std::span<T> weights, const T totalWeight, const Allocator& alloc = {}) {
    using EntryAllocator =
        std::allocator_traits<Allocator>::template rebind_alloc<wrs::AliasTableEntry<P, I>>;
    using Entry = wrs::AliasTableEntry<P, I>;

    const I N = weights.size();
    const P averageWeight = static_cast<P>(totalWeight) / static_cast<P>(N);
    I i = sweeping_internal::nextLight(weights, 0, averageWeight); // first light weight
    I j = sweeping_internal::nextHeavy(weights, 0, averageWeight);

    std::vector<Entry, EntryAllocator> aliasTable{N, EntryAllocator{alloc}};
    if (j == N) {
        fmt::println("ALL WEIGHTS ARE EQUAL");
        for (I k = 0; k < N; ++k) {
            /* std::numeric_limits<float>::i */
            aliasTable[k] = AliasTableEntry(P{1.0}, k);
        }
        return std::move(aliasTable);
    }
    assert(j < N);
    T w = weights[j]; // current heavy item
    while (j != N) {
        if (w > averageWeight) {
            // pack a light bucket!
            if (i >= N) {
                /* fmt::println("Bad exit: Packing remaining heavy"); */
                // no more light items. Pack last heavy bucket instead.

                // Assert that there the redisual weight fits perfectly
                // into the last heavy bucket!
                /* if (!(std::abs(w - averageWeight) < 1e-10 * totalWeight)) { */
                /*   fmt::println("w = {}, w! = {}", w, averageWeight); */
                /* } */
                /* assert(std::abs(averageWeight - w) < 1e-10 * totalWeight); */
                assert(j < N);
                aliasTable[j] = AliasTableEntry(P{1.0}, j);
                break;
            }
            assert(i < N);
            /* assert(weights[i] <= averageWeight); */
            P prob = static_cast<P>(weights[i]) / static_cast<P>(averageWeight);
            aliasTable[i] = AliasTableEntry(prob, j);
            w = (w + weights[i]) - averageWeight;
            i = sweeping_internal::nextLight(weights, i + 1, averageWeight);
        } else {
            // pack a heavy bucket!
            I jnext = sweeping_internal::nextHeavy(weights, j + 1, averageWeight);
            if (jnext == N) {
                /* fmt::println("Bad exit: Packing remaining light"); */
                // no more heavy items
                assert(j < N);
                /* if (!(std::abs(w - averageWeight) < 1e-8)) { */
                /*   fmt::println("w = {}, w! = {}", w, averageWeight); */
                /* } */
                /* assert(std::abs(w - averageWeight) < 1e-8); */
                aliasTable[j] = AliasTableEntry(P{1.0}, j);
                for (I k = i; k < N; ++k) {
                    /* assert(std::abs(weights[k] - averageWeight) < 1e-3); */
                    assert(k < N);
                    aliasTable[k] = AliasTableEntry(P{1.0}, k);
                }
                break;
            }
            P prob = static_cast<P>(w / averageWeight);
            assert(j < N);
            aliasTable[j] = AliasTableEntry(prob, jnext);
            j = jnext;
            w = (w + weights[j]) - averageWeight;
        }
    }
    return std::move(aliasTable);
}

namespace pmr {

template <wrs::arithmetic T, std::floating_point P, std::integral I>
wrs::pmr::AliasTable<P, I> sweeping_alias_table(
    std::span<T> weights, const T totalWeight, const std::pmr::polymorphic_allocator<void>& alloc) {
    // Hope for RTO
    return wrs::reference::sweeping_alias_table<T, P, I, std::pmr::polymorphic_allocator<void>>(
        weights, totalWeight, alloc);
}

} // namespace pmr

} // namespace wrs::reference
