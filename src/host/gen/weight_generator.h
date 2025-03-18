#pragma once

#include "src/host/why.hpp"
#include <cstdint>
#include <fmt/base.h>
#include <fmt/format.h>
#include <limits>
#include <random>
#include <spdlog/spdlog.h>
#include <vector>
namespace host {

enum class Distribution {
    UNIFORM,
    PSEUDO_RANDOM_UNIFORM,
    RANDOM_UNIFORM,
    SEEDED_RANDOM_UNIFORM,
    SEEDED_RANDOM_EXPONENTIAL,
    SEEDED_RANDOM_NORMAL,
    SEEDED_RANDOM_BRAUNIAN_NOISE,
    SEEDED_RANDOM_PERLIN_NOISE,
};

struct WeightGenInfo {
    Distribution distribution;
    uint32_t count;
};

std::string distribution_to_pretty_string(Distribution dist);

// The current implementation only works for floating point numbers
template <std::floating_point T = float, host::typed_allocator<T> Allocator = std::allocator<T>>
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
    case Distribution::SEEDED_RANDOM_EXPONENTIAL:
        loggingThreshold = 1e7;
        break;
    case Distribution::SEEDED_RANDOM_NORMAL:
        loggingThreshold = 1e7;
        break;
    default:
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
        std::mt19937 rng{3144847214312530820};
        std::uniform_real_distribution<T> dist{0.01f, 1.0f};
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
    case Distribution::SEEDED_RANDOM_EXPONENTIAL: {
        std::random_device seedRng{};
        std::uniform_int_distribution<uint64_t> seedDist{1, std::numeric_limits<uint64_t>::max()};
        uint64_t seed = seedDist(seedRng);
        SPDLOG_DEBUG(fmt::format("Seeding mt19937 with seed = {}", seed));
        std::mt19937 rng{seed};
        std::exponential_distribution<T> dist{1.0f};
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
    case Distribution::SEEDED_RANDOM_NORMAL: {
        std::random_device seedRng{};
        std::uniform_int_distribution<uint64_t> seedDist{1, std::numeric_limits<uint64_t>::max()};
        uint64_t seed = seedDist(seedRng);
        SPDLOG_DEBUG(fmt::format("Seeding mt19937 with seed = {}", seed));
        std::mt19937 rng{seed};
        std::normal_distribution<T> dist{5.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(count) * 100));
            }
            weights[i] = std::abs(dist(rng));
        }
        break;
    }
    case Distribution::SEEDED_RANDOM_BRAUNIAN_NOISE: {

        std::random_device seedRng{};
        std::uniform_int_distribution<uint64_t> seedDist{1, std::numeric_limits<uint64_t>::max()};
        uint64_t seed = seedDist(seedRng);
        SPDLOG_DEBUG(fmt::format("Seeding mt19937 with seed = {}", seed));
        std::mt19937 rng{seed};
        std::uniform_real_distribution<T> dist{-0.01f, 0.01f};
        T integrator = 0;
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(count) * 100));
            }
            integrator += dist(rng);
            integrator = std::clamp<T>(integrator, -1.0, 1.0);
            weights[i] = ((integrator + 1.0) / 2.0);
            /* fmt::println("{}", weights[i]); */
        }
        break;
    }
    case Distribution::SEEDED_RANDOM_PERLIN_NOISE: {
        // A standard permutation array for Perlin noise
        static int permutation[256] = {
            151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233, 7,   225,
            140, 36,  103, 30,  69,  142, 8,   99,  37,  240, 21,  10,  23,  190, 6,   148,
            247, 120, 234, 75,  0,   26,  197, 62,  94,  252, 219, 203, 117, 35,  11,  32,
            57,  177, 33,  88,  237, 149, 56,  87,  174, 20,  125, 136, 171, 168, 68,  175,
            74,  165, 71,  134, 139, 48,  27,  166, 77,  146, 158, 231, 83,  111, 229, 122,
            60,  211, 133, 230, 220, 105, 92,  41,  55,  46,  245, 40,  244, 102, 143, 54,
            65,  25,  63,  161, 1,   216, 80,  73,  209, 76,  132, 187, 208, 89,  18,  169,
            200, 196, 135, 130, 116, 188, 159, 86,  164, 100, 109, 198, 173, 186, 3,   64,
            52,  217, 226, 250, 124, 123, 5,   202, 38,  147, 118, 126, 255, 82,  85,  212,
            207, 206, 59,  227, 47,  16,  58,  17,  182, 189, 28,  42,  223, 183, 170, 213,
            119, 248, 152, 2,   44,  154, 163, 70,  221, 153, 101, 155, 167, 43,  172, 9,
            129, 22,  39,  253, 19,  98,  108, 110, 79,  113, 224, 232, 178, 185, 112, 104,
            218, 246, 97,  228, 251, 34,  242, 193, 238, 210, 144, 12,  191, 179, 162, 241,
            81,  51,  145, 235, 249, 14,  239, 107, 49,  192, 214, 31,  181, 199, 106, 157,
            184, 84,  204, 176, 115, 121, 50,  45,  127, 4,   150, 254, 138, 236, 205, 93,
            222, 114, 67,  29,  24,  72,  243, 141, 128, 195, 78,  66,  215, 61,  156, 180};

        // Generate a random seed for permutation shuffling
        std::random_device seedRng{};
        std::uniform_int_distribution<uint64_t> seedDist{1, std::numeric_limits<uint64_t>::max()};
        uint64_t seed = seedDist(seedRng);
        SPDLOG_DEBUG(fmt::format("Seeding mt19937 with seed = {}", seed));
        std::mt19937 rng{seed};

        // Shuffle the permutation array into a local array p of size 512
        static int p[512];
        {
            std::vector<int> permVec(std::begin(permutation), std::end(permutation));
            std::shuffle(permVec.begin(), permVec.end(), rng);

            for (int i = 0; i < 256; ++i) {
                p[i] = permVec[i];
                p[i + 256] = permVec[i];
            }
        }

        // Inline fade function (Perlin smoothing curve)
        auto fade = [&](float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); };

        // Inline grad function for 1D: returns x or -x based on hash bit
        auto grad = [&](int hash, float x) { return (hash & 1) == 0 ? x : -x; };

        // Inline 1D Perlin noise function
        auto perlin1D = [&](float x) -> float {
            // Determine which cell we're in
            int xi = static_cast<int>(std::floor(x)) & 255;
            float xf = x - std::floor(x);

            // Find hashed corners
            int a = p[xi];
            int b = p[xi + 1];

            // Smooth the fractional part
            float u = fade(xf);

            // Gradients for each corner
            float gradA = grad(a, xf);
            float gradB = grad(b, xf - 1.0f);

            // Linear interpolation
            return gradA + u * (gradB - gradA);
        };

        // Example frequency to control noise scale (adjust to taste)
        float frequency = 0.01f;

        // Fill the weights vector with 1D Perlin noise
        for (size_t i = 0; i < weights.size(); ++i) {
            if (enableLogging && nextChunk == i) {
                nextChunk += logChunkSize;
                SPDLOG_DEBUG(fmt::format("Generating numbers : {}% done",
                                         i / static_cast<float>(count) * 100));
            }
            float xCoord = static_cast<float>(i) * frequency;
            float noiseValue = perlin1D(xCoord);

            // Map from [-1, 1] to [0, 1]
            noiseValue = 0.5f * (noiseValue + 1.0f);

            // Assign to weights array
            weights[i] = noiseValue;
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
                                     const std::pmr::polymorphic_allocator<T>& alloc = {}) {
    return generate_weights<T, std::pmr::polymorphic_allocator<T>>(distribution, count, alloc);
}

}; // namespace pmr

} // namespace host
