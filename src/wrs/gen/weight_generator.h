#pragma once

#include "src/wrs/why.hpp"
#include <cstdint>
#include <fmt/base.h>
#include <fmt/format.h>
#include <limits>
#include <random>
#include <spdlog/spdlog.h>
#include <vector>
namespace wrs {

enum class Distribution {
    UNIFORM,
    PSEUDO_RANDOM_UNIFORM,
    RANDOM_UNIFORM,
    SEEDED_RANDOM_UNIFORM,
};

struct WeightGenInfo {
    Distribution distribution;
    uint32_t count;
};

std::string distribution_to_pretty_string(Distribution dist);

// The current implementation only works for floating point numbers
template <std::floating_point T = float, wrs::typed_allocator<T> Allocator = std::allocator<T>>
std::vector<T, Allocator>
generate_weights(const Distribution distribution, uint32_t count, const Allocator alloc = {}) {
    std::vector<T, Allocator> weights{count, alloc};

    size_t loggingThreshold = 1e10;
    switch (distribution) {
    case Distribution::UNIFORM:
        loggingThreshold = 1e7;
        break;
    case Distribution::PSEUDO_RANDOM_UNIFORM:
        loggingThreshold = 1e7;
        break;
    case Distribution::RANDOM_UNIFORM:
        loggingThreshold = 1e5;
        break;
    case Distribution::SEEDED_RANDOM_UNIFORM:
        loggingThreshold = 1e7;
        break;
    }

    bool enableLogging = count > loggingThreshold;
    constexpr size_t logCount = 10;
    size_t logChunkSize = count / logCount;
    size_t nextChunk = logChunkSize;

    switch (distribution) {
    case Distribution::UNIFORM: {
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(count) * 100));
            }
            weights[i] = 1.0f;
        }
        break;
    }
    case Distribution::PSEUDO_RANDOM_UNIFORM: {
        std::mt19937 rng{13500993188786726366ull};
        std::uniform_real_distribution<T> dist{0.0f, 1.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(count) * 100));
            }
            weights[i] = dist(rng);
        }
        break;
    }
    case Distribution::RANDOM_UNIFORM: {
        std::random_device rng{};
        std::uniform_real_distribution<T> dist{0.0f, 1.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(count) * 100));
            }
            weights[i] = dist(rng);
        }
        break;
    }
    case Distribution::SEEDED_RANDOM_UNIFORM: {
        std::random_device seedRng{};
        std::uniform_int_distribution<uint64_t> seedDist{1, std::numeric_limits<uint64_t>::max()};
        uint64_t seed = seedDist(seedRng);
        SPDLOG_DEBUG(fmt::format("Seeding mt19937 with seed = {}", seed));
        std::mt19937 rng{seed};
        std::uniform_real_distribution<T> dist{0.0f, 1.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(count) * 100));
            }
            weights[i] = dist(rng);
        }
        break;
    }
    }

    return weights;
}

namespace pmr {

template <std::floating_point T = float>
std::pmr::vector<T> generate_weights(const Distribution distribution,
                                     uint32_t count,
                                     const std::pmr::polymorphic_allocator<T>& alloc) {
    return wrs::generate_weights<T, std::pmr::polymorphic_allocator<T>>(distribution, count, alloc);
}

}; // namespace pmr

} // namespace wrs
