#include "./philox_eval.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "src/wrs/algorithm/prng/philox/Philox.hpp"
#include "src/wrs/algorithm/psa/construction/PSAC.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/types/glsl.hpp"
#include <fmt/base.h>
#include <random>
using namespace wrs;

void wrs::eval::write_philox_rmse_curve(const merian::ContextHandle& context) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    constexpr glsl::uint N = 1024 * 2048;
    constexpr std::size_t S = 1e10;

    wrs::Philox philox{context};

    constexpr glsl::uint MAX_SAMPLING_STEP_SIZE = 0x3FFFFFFF;
    constexpr glsl::uint SAMPLING_STEP_COUNT =
        (S + static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE) - 1) /
        static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE);
    constexpr glsl::uint RMSE_CURVE_TICKS = 100;
    constexpr glsl::uint SUBMIT_LIMIT = 4;

    wrs::Philox::Buffers local = wrs::Philox::Buffers::allocate(
        alloc, merian::MemoryMappingType::NONE, MAX_SAMPLING_STEP_SIZE);
    wrs::Philox::Buffers stage = wrs::Philox::Buffers::allocate(
        alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, MAX_SAMPLING_STEP_SIZE);

    std::vector<float> weights(N, 1.0f);
    merian::BufferHandle localWeights = alloc->createBuffer(
        wrs::PSAC::Buffers::WeightsLayout::size(N),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        merian::MemoryMappingType::NONE);
    merian::BufferHandle stageWeights = alloc->createBuffer(
        wrs::PSAC::Buffers::WeightsLayout::size(N), vk::BufferUsageFlagBits::eTransferSrc,
        merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    {
        vk::CommandBuffer cmd = cmdPool->create_and_begin();
        wrs::PSAC::Buffers::WeightsView stageView{stageWeights, N};
        wrs::PSAC::Buffers::WeightsView localView{localWeights, N};
        stageView.upload<float>(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
        cmd.end();
        queue->submit_wait(cmd);
    }

    std::mt19937 rng;
    std::uniform_int_distribution<glsl::uint> dist;

    constexpr bool USE_GPU_ACCELERATED_RMSE = true;
    if (USE_GPU_ACCELERATED_RMSE) {
        wrs::eval::RMSECurveAcceleratedBuilder rmseCurveBuilder{
            context, localWeights, static_cast<float>(N), N,
            wrs::eval::log10scale<uint64_t>(1000, S, RMSE_CURVE_TICKS)};

        std::vector<glsl::uint> samplesSection;
        std::size_t s = S;
        for (std::size_t i = 0; i < SAMPLING_STEP_COUNT;) {

            vk::CommandBuffer cmd = cmdPool->create_and_begin();

            std::size_t x = 0;
            while (i < SAMPLING_STEP_COUNT && x < SUBMIT_LIMIT) {

                std::size_t s2 = s;
                if (s2 == 0) {
                    continue;
                }
                if (s2 > MAX_SAMPLING_STEP_SIZE) {
                    s2 = MAX_SAMPLING_STEP_SIZE;
                }
                s -= MAX_SAMPLING_STEP_SIZE;

                glsl::uint seed = dist(rng);
                philox.run(cmd, local, s2, N, seed);

                rmseCurveBuilder.consume(cmd, local.samples, s2);
                ++i;
                ++x;
            }

            SPDLOG_INFO("Sectioned Sampling: {}/{} ~ {:.3}%", S - s, S,
                        100 * ((S - s) / static_cast<float>(S)));

            cmd.end();
            queue->submit_wait(cmd);
        }
        SPDLOG_INFO("Downloading from stage..");
        const auto& rmseCurve = rmseCurveBuilder.get();

        std::string path = "philox_rmse_curve.csv";
        SPDLOG_INFO("Writing curve to file \"{}\"...", path);
        wrs::exp::CSVWriter<2> csv{{"sample_size", "RMSE"}, path};
        for (const auto& [s, rmse] : rmseCurve) {
            csv.pushRow(s, rmse);
        }
    } else {

        wrs::eval::RMSECurveSectionedBuilder<double, float, glsl::uint> rmseCurveBuilder{
            weights, wrs::eval::log10scale<uint64_t>(1000, S, RMSE_CURVE_TICKS)};

        std::vector<glsl::uint> samplesSection;
        std::size_t s = S;
        for (std::size_t i = 0; i < SAMPLING_STEP_COUNT; ++i) {

            std::size_t s2 = s;
            if (s2 == 0) {
                continue;
            }
            if (s2 > MAX_SAMPLING_STEP_SIZE) {
                s2 = MAX_SAMPLING_STEP_SIZE;
            }
            s -= MAX_SAMPLING_STEP_SIZE;

            SPDLOG_INFO("Sectioned Sampling: {}/{} ~ {:.3}%", S - s, S,
                        100 * ((S - s) / static_cast<float>(S)));

            vk::CommandBuffer cmd = cmdPool->create_and_begin();

            glsl::uint seed = dist(rng);
            philox.run(cmd, local, s2, N, seed);


            wrs::Philox::Buffers::SamplesView stageView{stage.samples, s2};
            wrs::Philox::Buffers::SamplesView localView{local.samples, s2};
            localView.expectComputeWrite();         
            localView.copyTo(cmd, stageView);
            stageView.expectHostRead(cmd);

            cmd.end();
            queue->submit_wait(cmd);

            const auto& samples = stageView.download<glsl::uint>();

            rmseCurveBuilder.consume(samples);

        }
        SPDLOG_INFO("Downloading from stage..");
        const auto& rmseCurve = rmseCurveBuilder.get();

        std::string path = "philox_rmse_curve.csv";
        SPDLOG_INFO("Writing curve to file \"{}\"...", path);
        wrs::exp::CSVWriter<2> csv{{"sample_size", "RMSE"}, path};
        for (const auto& [s, rmse] : rmseCurve) {
            csv.pushRow(s, rmse);
        }
    }
}
