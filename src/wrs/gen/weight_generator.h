#pragma once

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

static std::string distribution_to_pretty_string(Distribution dist) {
    switch (dist) {
    case Distribution::UNIFORM:
        return "uniform";
    case Distribution::PSEUDO_RANDOM_UNIFORM:
        return "pseudo_random_uniform";
    case Distribution::RANDOM_UNIFORM:
        return "random_uniform";
    case Distribution::SEEDED_RANDOM_UNIFORM:
        return "seeded_random_uniform";
    default:
        return "NO-PRETTY-STRING-AVAIL";
    }
}

template <typename T = float, typename Allocator = std::allocator<T>>
static std::vector<T, Allocator> generate_weights(const Distribution distribution, uint32_t count,
    const Allocator alloc = {}) {
    std::vector<T, Allocator> weights(count, alloc);
    bool enableLogging = count > 1000000;
    constexpr size_t logCount = 10;
    size_t logChunkSize = count / logCount;
    size_t nextChunk = logChunkSize;

    switch (distribution) {
    case Distribution::UNIFORM: {
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(genInfo.count) * 100));
            }
            weights[i] = 1.0f;
        }
        break;
    }
    case Distribution::PSEUDO_RANDOM_UNIFORM: {
        std::mt19937 rng{2};
        std::uniform_real_distribution<T> dist{0.0f, 1.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(genInfo.count) * 100));
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
                                         i / static_cast<float>(genInfo.count) * 100));
            }
            weights[i] = dist(rng);
        }
        break;
    }
    case Distribution::SEEDED_RANDOM_UNIFORM:
        std::random_device seedRng{};
        std::uniform_int_distribution<uint64_t> seedDist{1, std::numeric_limits<uint64_t>::max()};
        std::mt19937 rng{seedDist(seedRng)};
        std::uniform_real_distribution<T> dist{0.0f, 1.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(genInfo.count) * 100));
            }
            weights[i] = dist(rng);
        }
        break;
    }
    return weights;
}

static std::vector<float> generate_weights(const WeightGenInfo& genInfo) {
    return generate_weights(genInfo.distribution, genInfo.count);
}

} // namespace wrs
