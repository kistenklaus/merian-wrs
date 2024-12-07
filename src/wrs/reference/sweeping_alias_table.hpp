#pragma once

#include "src/wrs/generic_types.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <memory>
#include <span>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vector>
namespace wrs::reference {

namespace sweeping_internal {

template <wrs::arithmetic T, std::floating_point P>
static std::size_t nextLight(const std::span<T>& weights, std::size_t i, P pivot) {
    if (i >= weights.size()) {
        return weights.size();
    }
    assert(i < weights.size());
    const auto it = std::find_if(weights.begin() + i, weights.end(),
                                 [pivot](const auto& w) { return w <= pivot; });
    if (it == weights.end()) {
        return weights.size();
    } else {
        return static_cast<std::size_t>(it - weights.begin());
    }
}
template <wrs::arithmetic T, std::floating_point P>
static std::size_t nextHeavy(const std::span<T>& weights, std::size_t i, P pivot) {
    if (i >= weights.size()) {
        return weights.size();
    }
    assert(i < weights.size());
    const auto it = std::find_if(weights.begin() + i, weights.end(),
                                 [pivot](const auto& w) { return w > pivot; });
    if (it == weights.end()) {
        return weights.size();
    } else {
        return static_cast<std::size_t>(it - weights.begin());
    }
}

} // namespace sweeping_internal

template <wrs::arithmetic T, std::floating_point P, wrs::generic_allocator Allocator = std::allocator<void>>
std::vector<wrs::alias_table_entry_t<P>,
            typename std::allocator_traits<Allocator>::template rebind_alloc<
                wrs::alias_table_entry_t<P>>>
sweeping_alias_table(std::span<T> weights, const T totalWeight, const Allocator& alloc = {}) {
    using EntryAllocator = std::allocator_traits<Allocator>::template rebind_alloc<
        wrs::alias_table_entry_t<P>>;
    using Entry = wrs::alias_table_entry_t<P>;

    const std::size_t N = weights.size();
    const P averageWeight = static_cast<P>(totalWeight) / static_cast<P>(N);
    std::size_t i = sweeping_internal::nextLight(weights, 0, averageWeight); // first light weight
    std::size_t j = sweeping_internal::nextHeavy(weights, 0, averageWeight);

    std::vector<Entry, EntryAllocator> aliasTable{N, EntryAllocator{alloc}};
    if (j == N) {
        for (std::size_t k = 0; k < N; ++k) {
            /* std::numeric_limits<float>::i */
            aliasTable[k] = std::make_tuple(static_cast<P>(1.0), k);
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
                aliasTable[j] = std::make_tuple(1.0f, j);
                break;
            }
            assert(i < N);
            /* assert(weights[i] <= averageWeight); */
            P prob = static_cast<P>(weights[i]) / static_cast<P>(averageWeight);
            aliasTable[i] = std::make_tuple(prob, j);
            w = (w + weights[i]) - averageWeight;
            i = sweeping_internal::nextLight(weights, i + 1, averageWeight);
        } else {
            // pack a heavy bucket!
            std::size_t jnext = sweeping_internal::nextHeavy(weights, j + 1, averageWeight);
            if (jnext == N) {
                /* fmt::println("Bad exit: Packing remaining light"); */
                // no more heavy items
                assert(j < N);
                /* if (!(std::abs(w - averageWeight) < 1e-8)) { */
                /*   fmt::println("w = {}, w! = {}", w, averageWeight); */
                /* } */
                /* assert(std::abs(w - averageWeight) < 1e-8); */
                aliasTable[j] = std::make_tuple(1.0f, j);
                for (size_t k = i; k < N; ++k) {
                    /* assert(std::abs(weights[k] - averageWeight) < 1e-3); */
                    assert(k < N);
                    aliasTable[k] = std::make_tuple(1.0f, k);
                }
                break;
            }
            P prob = static_cast<P>(w / averageWeight);
            assert(j < N);
            aliasTable[j] = std::make_tuple(prob, jnext);
            j = jnext;
            w = (w + weights[j]) - averageWeight;
        }
    }
    return std::move(aliasTable);
}

namespace pmr {

template <wrs::arithmetic T, std::floating_point P>
std::vector<wrs::alias_table_entry_t<P>,
            std::pmr::polymorphic_allocator<wrs::alias_table_entry_t<P>>>
sweeping_alias_table(std::span<T> weights,
                     const T totalWeight,
                     const std::pmr::polymorphic_allocator<void>& alloc) {
    // Hope for RTO
    return wrs::reference::sweeping_alias_table<T, P, std::pmr::polymorphic_allocator<void>>(
        weights, totalWeight, alloc);
}

} // namespace pmr

} // namespace wrs::reference
