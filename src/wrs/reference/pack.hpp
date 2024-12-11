#pragma once

#include "src/wrs/generic_types.hpp"
#include "src/wrs/why.hpp"
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <tuple>
namespace wrs::reference {

template <wrs::arithmetic T, std::floating_point P, std::integral I>
P pack(const std::span<I> heavyIndices,
       const std::span<I> lightIndicies,
       const std::span<T> weights,
       const P averageWeight,
       const I i0,
       const I i1,
       const I j0,
       const I j1,
       const T spill,
       std::span<wrs::alias_table_entry_t<P, I>> aliasTable) {
    const I N = static_cast<I>(weights.size());
    assert(static_cast<I>(aliasTable.size()) == N);
    const I heavyCount = static_cast<I>(heavyIndices.size());
    const I lightCount = static_cast<I>(lightIndicies.size());
    /* fmt::println("weights = {}", weights); */
    /* fmt::println("heavyIndices = {}", heavyIndices); */
    /* fmt::println("lightIndices = {}", lightIndicies); */
    /* fmt::println("averageWeight = {}", averageWeight); */
    /* fmt::println("i0 = {}, i1 = {}, j0 = {}, j1 = {}, s = {}", i0, i1, j0, j1, spill); */
    I i = i0;
    // Can underflow on the first iteration!, which is fine because it overflows right after
    I j = j0;
    double w = static_cast<P>(spill);
    if (w == 0.0f) {
        assert(j < heavyCount);
        I h = heavyIndices[j];
        assert(h < N);
        w = weights[h];
    }
    uint64_t lightToPack = i1 - static_cast<int64_t>(i);
    uint64_t heavyToPack = j1 - static_cast<int64_t>(j);
    while (j != heavyCount) {
        /* fmt::println("TODO-list : light = {}, heavy = {}", lightToPack, heavyToPack); */
        /* fmt::println("w = {}, i = {}, j = {}", w, i, j); */
        /* auto diff = std::abs(w - averageWeight); */
        /* if (diff < 0.01) { */
        /*     SPDLOG_WARN(fmt::format("uhh {}", diff)); */
        /* } */
        if (w > averageWeight) {
            if (i >= i1) {
                /* fmt::println("Exit while packing light"); */
                if (j != j1) {
                    // Pack remaining heavy
                    /* SPDLOG_WARN("Packing remaining heavy"); */
                    /* if (std::abs(averageWeight - w) > 0.001) { */
                    /*     SPDLOG_WARN(fmt::format("Numerical instability : Expected : {}, Got :
                     * {}", */
                    /*                             averageWeight, w)); */
                    /* } */
                    assert(j < heavyCount);
                    while (j < j1) {
                        I h = heavyIndices[j];
                        assert(h < N);
                        aliasTable[h] = std::make_tuple(1.0f, h);
                        heavyToPack -= 1;
                        w = (w + 0) - averageWeight;
                        j += 1;
                    }
                }
                break;
            }
            // pack light bucket!
            assert(i < lightCount);
            I l = lightIndicies[i];
            assert(j < heavyCount);
            I h = heavyIndices[j];

            assert(l < N);
            assert(h < N);

            /* fmt::println("\tPacking light element {} at {}", weights[l], l); */
            P prob = weights[l] / averageWeight; // normalize to redirect prob.
            /* fmt::println("\tRedirecting access weight to {} with probability {}", h, prob); */
            aliasTable[l] = std::make_tuple(prob, h);
            lightToPack -= 1;
            /* fmt::println("\tpacking {}", l); */
            w = (w + weights[l]) - averageWeight;
            i += 1; // next light item
        } else {
            assert(j < heavyCount);
            I h = heavyIndices[j];
            assert(h < N);
            // pack heavy element
            if (j >= j1) {
                /* assert(j == j1); // Just assuming for now! */
                // no more heavy elements pack remaining light
                /* fmt::println("Exit while packing heavy, lm = {}, hm = {}", lightToPack, */
                /*              heavyToPack); */
                /* fmt::println("w = {}, avg = {}", w, averageWeight); */
                while (i < i1) {
                    // pack missing light elements
                    I l = lightIndicies[i];
                    I h = heavyIndices[j];
                    P prob = weights[l] / averageWeight;
                    aliasTable[l] = std::make_tuple(prob, h);
                    lightToPack -= 1;
                    w = (w + weights[l]) - averageWeight;
                    ++i;
                    /* fmt::println("\tpacking badly {} residual weight = {}", l, w); */
                }
                break;
            }
            P prob = w / averageWeight;
            if (j + 1 >= heavyCount) {
                // This should only happen on the last heavy element!
                aliasTable[h] = std::make_tuple(prob, h);
                heavyToPack -= 1;
                /* fmt::println("\tpacking {} (last heavy element)", h); */
                w = (w + 0) - averageWeight;
                while (i < i1) {
                    // pack missing light elements
                    I l = lightIndicies[i];
                    I h = heavyIndices[j];
                    P prob = weights[l] / averageWeight;
                    aliasTable[l] = std::make_tuple(prob, h);
                    lightToPack -= 1;
                    w = (w + weights[l]) - averageWeight;
                    ++i;
                    /* fmt::println("\tpacking badly {} residual weight = {}", l, w); */
                }
                break;
            } else {
                I hnext = heavyIndices[j + 1];
                assert(hnext < N);
                aliasTable[h] = std::make_tuple(prob, hnext);
                heavyToPack -= 1;
                /* fmt::println("\tpacking {}", h); */
                w = (w + weights[hnext]) - averageWeight;
                j += 1; // next heavy item
            }
        }
    }
    /* assert(lightToPack == 0); */
    /* assert(heavyToPack == 0); */
    /* static int count = 0; */
    /* if (lightToPack != 0 || heavyToPack != 0) { */
    /*     count += 1; */
    /*     fmt::println("({},{},{},{},{}), remaining = ({},{}) [{}]", i0, i1, j0, j1, spill, */
    /*                  lightToPack, heavyToPack, count); */
    /* } */
    /* fmt::println("Redisual : {}", w); */
    return w;
}

template <wrs::arithmetic T,
          std::floating_point P,
          std::integral I,
          wrs::typed_allocator<wrs::alias_table_entry_t<P, I>> Allocator>
wrs::alias_table_t<P, I, Allocator> packSplits(const std::span<I> heavyIndices,
                                               const std::span<I> lightIndicies,
                                               const std::span<T> weights,
                                               const T averageWeight,
                                               std::span<wrs::split_t<T, I>> splits,
                                               const Allocator& alloc = {}) {
    wrs::split_t<T, I> prevSplit = std::make_tuple(I{}, I{}, T{});
    const I N = static_cast<I>(weights.size());
    wrs::alias_table_t<P, I, Allocator> aliasTable{N, alloc};
    for (size_t k = 0; k < splits.size(); k++) {
        const auto& [i0, j0, s] = prevSplit;
        const auto& [i1, j1, r] = splits[k];
        P residual = wrs::reference::pack<T, P, I>(heavyIndices, lightIndicies, weights,
                                                   averageWeight, i0, i1, j0, j1, s, aliasTable);
        /* if (std::abs(residual - r) > 0.25) { */
        /*     SPDLOG_WARN( */
        /*         fmt::format("Numerical instability. Expected residual {}, Got {}", r, residual)); */
        /* } */
        prevSplit = splits[k];
    }

    return aliasTable; // NRTO
}

namespace pmr {
template <wrs::arithmetic T, std::floating_point P, std::integral I>
wrs::pmr::alias_table_t<P, I>
packSplits(const std::span<I>& heavyIndices,
           const std::span<I>& lightIndices,
           const std::span<T>& weights,
           const T averageWeight,
           const std::span<wrs::split_t<T, I>> splits,
           const std::pmr::polymorphic_allocator<wrs::alias_table_entry_t<P, I>>& alloc = {}) {
    return wrs::reference::packSplits<
        T, P, I, std::pmr::polymorphic_allocator<wrs::alias_table_entry_t<P, I>>>(
        heavyIndices, lightIndices, weights, averageWeight, splits, alloc); // URTO
}

} // namespace pmr

} // namespace wrs::reference
