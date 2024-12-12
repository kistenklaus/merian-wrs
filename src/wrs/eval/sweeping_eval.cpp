#include "./sweeping_eval.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/sample_alias_table.hpp"
#include "src/wrs/reference/sweeping_alias_table.hpp"
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

using I = std::size_t;

template <typename T>
static std::pmr::vector<std::tuple<I, T>>
sweeping_rmse_curve(EvalCase eval,
               std::size_t S,
               wrs::eval::log10::IntLogScaleRange<I> scale,
               std::pmr::memory_resource* resource) {
    auto weights = wrs::pmr::generate_weights<T>(eval.distribution, eval.weightCount, resource);

    auto totalWeight = wrs::reference::kahan_reduction<T>(weights);

    SPDLOG_DEBUG("Constructing alias table");

    const auto aliasTable =
      wrs::reference::pmr::sweeping_alias_table<T, T, I>(weights, totalWeight, resource);

    SPDLOG_DEBUG("Sampling alias table (this might take while");
    const auto samples = wrs::reference::pmr::sample_alias_table<T, I>(aliasTable, S, resource);

    auto rmseCurve =
        wrs::eval::pmr::rmse_curve<T, I>(weights, samples, scale, std::nullopt, resource);
    return rmseCurve;
}

void wrs::eval::write_sweeping_rmse_curves() {

    SPDLOG_INFO("Writing Sweeping RMSE curves to csv file (this may take a while)");

    const EvalCase EVAL_CASES[] = {
        //
        {
            .name = "float-1024x2048-uniform",
            .weightCount = 1024 * 2048,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_FLOAT,
        },
        {
            .name = "double-1024x2048-uniform",
            .weightCount = 1024 * 2048,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_DOUBLE,
        },
        {
            .name = "float-1024x2048-normal",
            .weightCount = 1024 * 2048,
            .distribution = Distribution::SEEDED_RANDOM_NORMAL,
            .weightType = WEIGHT_TYPE_FLOAT,
        },
        {
            .name = "double-1024x2048-normal",
            .weightCount = 1024 * 2048,
            .distribution = Distribution::SEEDED_RANDOM_NORMAL,
            .weightType = WEIGHT_TYPE_DOUBLE,
        },
        {
            .name = "float-1024x2048-exponential",
            .weightCount = 1024 * 2048,
            .distribution = Distribution::SEEDED_RANDOM_EXPONENTIAL,
            .weightType = WEIGHT_TYPE_FLOAT,
        },
        {
            .name = "double-1024x2048-exponential",
            .weightCount = 1024 * 2048,
            .distribution = Distribution::SEEDED_RANDOM_EXPONENTIAL,
            .weightType = WEIGHT_TYPE_DOUBLE,
        },
        {
            .name = "float-1024x128-uniform",
            .weightCount = 1024 * 128,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_FLOAT,
        },
        {
            .name = "double-1024x128-uniform",
            .weightCount = 1024 * 128,
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .weightType = WEIGHT_TYPE_DOUBLE,
        },
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
            auto curve = sweeping_rmse_curve<float>(eval, S, scale, &resource);
            for (std::size_t j = 0; j < curve.size(); ++j) {
                curves[j][i] = static_cast<double>(std::get<1>(curve[j]));
            }
            break;
        }
        case WEIGHT_TYPE_DOUBLE: {
            auto curve = sweeping_rmse_curve<double>(eval, S, scale, &resource);
            for (std::size_t j = 0; j < curve.size(); ++j) {
                curves[j][i] = std::get<1>(curve[j]);
            }
            break;
        }
        }
        stackResource.reset();
        ++i;
    }

    const std::string csvPath = "./sweeping_reference_rmse.csv";
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
