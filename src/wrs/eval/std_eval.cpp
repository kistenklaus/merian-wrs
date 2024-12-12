#include "./std_eval.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include <cmath>
#include <random>
#include <spdlog/spdlog.h>

enum WeightType {
    WEIGHT_TYPE_FLOAT,
    WEIGHT_TYPE_DOUBLE,
};

struct EvalCase {
    std::string name;
    std::size_t weightCount;
    wrs::Distribution distribution;
    WeightType weightType;
};

using I = uint32_t;

template <typename T>
static std::pmr::vector<std::tuple<I, T>>
std_rmse_curve(EvalCase eval,
               std::size_t S,
               wrs::eval::log10::IntLogScaleRange<I> scale,
               std::pmr::memory_resource* resource) {

    auto weights = wrs::pmr::generate_weights<T>(eval.distribution, eval.weightCount, resource);

    std::random_device seedRng;
    std::uniform_int_distribution<uint64_t> seedDist;

    std::mt19937 rng{seedDist(seedRng)};

    SPDLOG_DEBUG("Constructing distribution");

    std::discrete_distribution<I> dist{weights.begin(), weights.end()};
    /* std::uniform_int_distribution<I> dist{0, static_cast<I>(weights.size() - 1)}; */

    SPDLOG_DEBUG("Sampling from distribution... (this may taka while)");

    std::pmr::vector<I> samples{S, resource};
    std::size_t logStep = S / 100;
    std::size_t logThresh = logStep;
    for (size_t i = 0; i < S; ++i) {
        if (i == logThresh) {
            logThresh += logStep;
            SPDLOG_DEBUG(fmt::format("Sampling in progess {}%",
                                     std::round(i * 100 / static_cast<float>(S))));
        }
        samples[i] = dist(rng);
    }
    SPDLOG_DEBUG("Computing RMSE curve...");
    return wrs::eval::pmr::rmse_curve<T, I>(weights, samples, scale, std::nullopt, resource);
}

void wrs::eval::write_std_rmse_curves() {

    SPDLOG_INFO(
        "Writing std::discrete_distribution RMSE curves to csv file (this may take a while)");

    const EvalCase EVAL_CASES[] = {
        //
        {
            .name = "float-1024-uniform",
            .weightCount = 1024,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_FLOAT,
        },
        {
            .name = "double-1024-uniform",
            .weightCount = 1024,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_DOUBLE,
        },
        {
            .name = "float-2048-uniform",
            .weightCount = 2048,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_FLOAT,
        },
        {
            .name = "double-2048-uniform",
            .weightCount = 2048,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_DOUBLE,
        },
        {
            .name = "float-1024x2048-uniform",
            .weightCount = 1024 * 2048,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_FLOAT,
        },
    };
    constexpr std::size_t CASE_COUNT = sizeof(EVAL_CASES) / sizeof(EvalCase);
    constexpr std::size_t S = 1e9;
    constexpr std::size_t ticks = 500;

    wrs::eval::log10::IntLogScaleRange<I> scale = wrs::eval::log10scale<I>(1, S, ticks);

    wrs::memory::StackResource stackResource{10000 * sizeof(float)};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource resource{&fallbackResource};

    std::vector<std::array<double, CASE_COUNT>> curves{ticks};

    std::size_t i = 0;
    std::array<std::string, CASE_COUNT + 1> caseNames;
    caseNames[0] = "sample_size";
    for (const auto& eval : EVAL_CASES) {
        SPDLOG_INFO(fmt::format("Computing RMSE for evaluation case {}", eval.name));
        caseNames[i + 1] = eval.name;
        switch (eval.weightType) {
        case WEIGHT_TYPE_FLOAT: {
            auto curve = std_rmse_curve<float>(eval, S, scale, &resource);
            for (std::size_t j = 0; j < curve.size(); ++j) {
                curves[j][i] = static_cast<double>(std::get<1>(curve[j]));
            }
            break;
        }
        case WEIGHT_TYPE_DOUBLE: {
            auto curve = std_rmse_curve<double>(eval, S, scale, &resource);
            for (std::size_t j = 0; j < curve.size(); ++j) {
                curves[j][i] = std::get<1>(curve[j]);
            }
            break;
        }
        }
        stackResource.reset();
        ++i;
    }

    const std::string csvPath = "./std_rmse.csv";
    SPDLOG_INFO(fmt::format("Combining computed RSME curves and writing results to {}", csvPath));
    wrs::exp::CSVWriter<CASE_COUNT + 1> csv{caseNames, csvPath};
    auto scaleIt = scale.begin();
    for (std::size_t t = 0; t < curves.size(); ++t, ++scaleIt) {
        csv.unsafePushValue(*scaleIt, false);
        for (std::size_t r = 0; r < CASE_COUNT; r++) {
            csv.unsafePushValue(curves[t][r], r == CASE_COUNT - 1);
        }
        csv.unsafeEndRow();
    }
}
