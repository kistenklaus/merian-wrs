#pragma once

#include "src/host/types/alias_table.hpp"
#include "src/host/types/split.hpp"
#include "src/host/why.hpp"
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <fmt/base.h>
#include <spdlog/spdlog.h>

namespace host::reference {

template <arithmetic T, std::floating_point P, std::integral I>
P pack2(const std::span<const I> heavyIndices,
        const std::span<const I> lightIndices,
        const std::span<const T> weights,
        const P averageWeight,
        const I i0,
        const I i1,
        const I j0,
        const I j1,
        const T spill,
        std::span<AliasTableEntry<P, I>> aliasTable) {
    const I N = static_cast<I>(weights.size());
    assert(static_cast<I>(aliasTable.size()) == N);
    const I heavyCount = static_cast<I>(heavyIndices.size());
    I i = i0;
    I j = j0;
    P w = static_cast<P>(spill);

    if (w == 0.0f) {
        assert(j < heavyCount);
        I h = heavyIndices[j];
        assert(h < N);
        w = weights[h];
    }

    while (i != i1 || j != j1) {
        bool packHeavy;
        if (j == j1) {
            packHeavy = false;
        } else if (i == i1) {
            packHeavy = true;
        } else {
            packHeavy = w <= averageWeight;
        }

        I h = heavyIndices[j];
        I weightIdx;
        if (packHeavy) {
            weightIdx = heavyIndices[j + 1];
        } else {
            weightIdx = lightIndices[i];
        }
        P weight = weights[weightIdx];
        P p;
        I a;
        I idx;
        if (packHeavy) {
            p = w / averageWeight;
            a = weightIdx;
            idx = h;
            // fmt::println("Packing heavy {}", j);
            j += 1;
        } else {
            p = weight / averageWeight;
            a = h;
            idx = weightIdx;
            // fmt::println("Packing light {}", i);
            i += 1;
        }

        aliasTable[idx] = AliasTableEntry(p, a);
        w = (w + weight) - averageWeight;
    }
    // fmt::println("it = {}", it);
    if (j1 == heavyCount - 1) { // last bucket!
        I h = heavyIndices[j];
        aliasTable[h] = AliasTableEntry(P{1.0}, h);
    }
    return w;
}

template <arithmetic T, std::floating_point P, std::integral I>
P pack(std::span<const I> heavyIndices,
       std::span<const I> lightIndices,
       std::span<const T> weights,
       const P averageWeight,
       const I i0,
       const I i1,
       const I j0,
       const I j1,
       const T spill,
       std::span<AliasTableEntry<P, I>> aliasTable) {
    const I N = static_cast<I>(weights.size());
    assert(static_cast<I>(aliasTable.size()) == N);
    const I heavyCount = static_cast<I>(heavyIndices.size());
    I i = i0;
    I j = j0;
    double w = static_cast<P>(spill);

    if (w == 0.0f) {
        assert(j < heavyCount);
        I h = heavyIndices[j];
        assert(h < N);
        w = weights[h];
    }

    while (j != heavyCount) {
        if (w > averageWeight) {
            if (i >= i1) {
                if (j != j1) {
                    while (j < j1) {
                        I h = heavyIndices[j];
                        aliasTable[h] = AliasTableEntry(T{1.0}, h);
                        w = (w + 0) - averageWeight;
                        j += 1;
                    }
                }
                break;
            }
            // pack light bucket!
            I l = lightIndices[i];
            I h = heavyIndices[j];
            P prob = weights[l] / averageWeight; // normalize to redirect prob.
            aliasTable[l] = AliasTableEntry(prob, h);
            w = (w + weights[l]) - averageWeight;
            i += 1; // next light item
        } else {
            I h = heavyIndices[j];
            if (j >= j1) {
                while (i < i1) {
                    // pack missing light elements
                    I l = lightIndices[i];
                    I h = heavyIndices[j];
                    P prob = weights[l] / averageWeight;
                    aliasTable[l] = AliasTableEntry(prob, h);
                    w = (w + weights[l]) - averageWeight;
                    i += 1;
                }
                break;
            }
            P prob = w / averageWeight;
            if (j + 1 >= heavyCount) {
                // This should only happen on the last heavy element!
                aliasTable[h] = AliasTableEntry(prob, h);
                w = (w + 0) - averageWeight;
                while (i < i1) {
                    // pack missing light elements
                    I l = lightIndices[i];
                    I h = heavyIndices[j];
                    P prob = weights[l] / averageWeight;
                    aliasTable[l] = AliasTableEntry(prob, h);
                    w = (w + weights[l]) - averageWeight;
                    i += 1;
                }
                break;
            } else {
                I hnext = heavyIndices[j + 1];
                aliasTable[h] = AliasTableEntry(prob, hnext);
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

template <arithmetic T,
          std::floating_point P,
          std::integral I,
          typed_allocator<AliasTableEntry<P, I>> Allocator>
AliasTable<P, I, Allocator> packSplits(std::span<const I> heavyIndices,
                                       std::span<const I> lightIndicies,
                                       std::span<const T> weights,
                                       T averageWeight,
                                       std::span<const Split<T, I>> splits,
                                       const Allocator& alloc = {}) {
    Split<T, I> prevSplit;
    const I N = static_cast<I>(weights.size());
    AliasTable<P, I, Allocator> aliasTable{N, alloc};
    for (size_t k = 0; k < splits.size(); k++) {
        const Split<T, I> split = splits[k];
        /* fmt::println("k = {} : ({},{},{})-({},{},{})", k, prevSplit.i, prevSplit.j,
         * prevSplit.spill, split.i, split.j, split.spill); */
        reference::pack2<T, P, I>(heavyIndices, lightIndicies, weights, averageWeight, prevSplit.i,
                                  split.i, prevSplit.j, split.j, prevSplit.spill, aliasTable);
        prevSplit = split;
    }

    return aliasTable; // NRTO
}

namespace pmr {
template <arithmetic T, std::floating_point P, std::integral I>
host::pmr::AliasTable<P, I>
packSplits(std::span<const I> heavyIndices,
           std::span<const I> lightIndices,
           std::span<const T> weights,
           const T averageWeight,
           std::span<const Split<T, I>> splits,
           const std::pmr::polymorphic_allocator<AliasTableEntry<P, I>>& alloc = {}) {
    return reference::packSplits<T, P, I, std::pmr::polymorphic_allocator<AliasTableEntry<P, I>>>(
        heavyIndices, lightIndices, weights, averageWeight, splits, alloc); // URTO
}

} // namespace pmr

} // namespace host::reference
