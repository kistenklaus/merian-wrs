#include "./hst_std_eval.hpp"
#include "src/wrs/algorithm/hs/HSTRepr.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/gen/weight_generator.h"
#include <cstddef>
#include <fmt/base.h>
#include <random>
#include <ranges>
#include <span>
#include <vector>

using namespace wrs;
constexpr std::size_t N = 128;
constexpr Distribution DIST = wrs::Distribution::UNIFORM;
constexpr std::size_t S = 1024 * 2048;

void sample(std::span<const float> weights, std::size_t s) {

    std::vector<float> hst{weights.begin(), weights.end()};
    hst::HSTRepr repr{weights.size()};
    hst.resize(repr.size());
    for (const auto& level : repr.get()) {
        for (glsl::uint i = 0; i < level.numParents; ++i) {
            float w = hst[level.childOffset + i * 2 + 0] + hst[level.childOffset + i * 2 + 1];
            hst[level.parentOffset + i] = w;
        }
    }

    for (std::size_t i = 0; i < hst.size(); ++i) {
      fmt::println("[{}]: {}", i, hst[i]);
    }

    std::mt19937 rng{1234u};

    std::vector<glsl::uint> histogram(repr.size() + 1);
    histogram[repr.size()] = s;
    glsl::uint active = 1;
    glsl::uint parentOffset = repr.size();
    for (const auto& level : repr.get() | std::ranges::views::reverse) {
      fmt::println("{}", parentOffset);
      for (glsl::uint i = 0; i < active; ++i) {
        glsl::uint n = histogram[parentOffset + i];
        float w1 = hst[level.parentOffset + 2 * i + 0];
        float w2 = hst[level.parentOffset + 2 * i + 1];
        float p = w1 / (w1 + w2);
        std::binomial_distribution<glsl::uint> dist{n, p};
        glsl::uint s = dist(rng);
        histogram[level.parentOffset + 2 * i + 0] = s;
        histogram[level.parentOffset + 2 * i + 1] = n - s;
      }
      active = level.numParents;
      parentOffset = level.parentOffset;
    }

    {
      glsl::uint parentOffset = repr.get().front().parentOffset;
      glsl::uint childOffset = repr.get().front().childOffset;
      for (glsl::uint i = 0; i < repr.get().front().numParents; ++i) {
        glsl::uint n = histogram[parentOffset + i];
        float w1 = hst[childOffset + 2 * i + 0];
        float w2 = hst[childOffset + 2 * i + 1];
        float p = w1 / (w1 + w2);
        std::binomial_distribution<glsl::uint> dist{n, p};
        glsl::uint s = dist(rng);
        histogram[childOffset + 2 * i + 0] = s;
        histogram[childOffset + 2 * i + 1] = n - s;
      }
    }

    for (std::size_t i = 0; i < histogram.size(); ++i) {
      fmt::println("[{}]: {}", i, histogram[i]);
    }

}

void wrs::eval::write_hst_std_rmse_curves() {

    std::vector<float> weights = wrs::generate_weights(DIST, N);

    sample(weights, S);
}
