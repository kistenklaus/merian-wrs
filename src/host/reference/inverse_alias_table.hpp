#pragma once

#include "src/host/reference/reduce.hpp"
#include "src/host/types/alias_table.hpp"
#include "src/host/why.hpp"
#include <algorithm>
#include <concepts>
#include <memory>
namespace host::reference {

template <std::floating_point P,
          std::integral I,
          host::typed_allocator<P> Allocator = std::allocator<P>>
std::vector<P, Allocator>
alias_table_to_normalized_weights(ImmutableAliasTableReference<P, I> aliasTable,
                                  const Allocator& alloc = {}) {

    std::vector<P, Allocator> relativeProb{aliasTable.size(), alloc};

    for (std::size_t i = 0; i < aliasTable.size(); ++i) {
        const AliasTableEntry<P, I>& entry = aliasTable[i];
        const float p = std::clamp<P>(entry.p, P(0.0), P(1.0));
        relativeProb[i] += p;
        relativeProb[entry.a] += 1.0 - p;
    }
    return relativeProb;
}

template <std::floating_point P, typed_allocator<P> Allocator = std::allocator<P>>
std::vector<P, Allocator> normalize_weights(std::span<const P> weights,
                                            const Allocator& alloc = {}) {
    const P sum = reference::reduce<P, Allocator>(weights, alloc);
    P mean = sum / weights.size();
    std::vector<P, Allocator> normalized{weights.size(), alloc};
    for (std::size_t i = 0; i < weights.size(); ++i) {
        normalized[i] = weights[i] / mean;
    }
    return normalized;
}

template <std::floating_point W,
          std::floating_point P,
          typed_allocator<P> Allocator = std::allocator<P>>
std::vector<P, Allocator> weights_to_probabilities(std::span<const W> weights,
                                                   const Allocator& alloc = {}) {
    using WAllocator = std::allocator_traits<Allocator>::template rebind_alloc<W>;
    const W sum = reference::reduce<W, WAllocator>(weights, WAllocator(alloc));
    std::vector<P, Allocator> probs{weights.size(), alloc};
    for (std::size_t i = 0; i < weights.size(); ++i) {
        probs[i] = static_cast<P>(weights[i]) / static_cast<P>(sum);
    }
    return probs;
}

} // namespace host::reference
