#pragma once

#include "src/wrs/generic_types.hpp"
#include "src/wrs/why.hpp"
#include <fmt/base.h>
#include <random>
#include <vector>
namespace wrs::reference {

static std::size_t C = 0;

namespace alias_table_internals {


template <std::floating_point P, std::integral I>
I sample_single(const wrs::const_alias_table_ref<P, I> aliasTable, const I u1, const float u2) {
    assert(u1 < aliasTable.size());
    const auto& [p, a] = aliasTable[u1];
    if (u2 > p) {
        return a;
    } else {
        return u1;
    }
}

} // namespace alias_table_internals

template <std::floating_point P, std::integral I>
void sample_alias_table_inplace(const wrs::const_alias_table_ref<P, I> aliasTable,
                                std::span<I> out_samples) {
    /* fmt::println("SampleCount = {}", out_samples); */
    std::random_device truRng;
    std::uniform_int_distribution<uint64_t> seedDist;
    std::mt19937 rng {seedDist(truRng)};
    std::uniform_int_distribution<I> u1Dist{0, aliasTable.size() - 1};
    std::uniform_real_distribution<P> u2Dist{0.0, 1.0};
    for (size_t i = 0; i < out_samples.size(); ++i) {
        const I u1 = u1Dist(rng);
        const P u2 = u2Dist(rng);
        out_samples[i] = alias_table_internals::sample_single<P, I>(aliasTable, u1, u2);
    }
}

template <std::floating_point P,
          std::integral I,
          wrs::typed_allocator<I> Allocator = std::allocator<I>>
std::vector<I, Allocator> sample_alias_table(const wrs::const_alias_table_ref<P, I> aliasTable,
                                             std::size_t S,
                                             const Allocator& alloc) {
    std::vector<I, Allocator> samples{S, alloc};
    sample_alias_table_inplace<P, I>(aliasTable, samples);
    return samples;
}

namespace pmr {

template <std::floating_point P, std::integral I>
std::pmr::vector<I> sample_alias_table(const wrs::const_alias_table_ref<P, I> aliasTable,
                                       std::size_t S,
                                       const std::pmr::polymorphic_allocator<I>& alloc = {}) {
    return wrs::reference::sample_alias_table<P, I>(aliasTable, S, alloc);
}

} // namespace pmr

} // namespace wrs::reference
