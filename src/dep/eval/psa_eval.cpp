#include "./psa_eval.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_shaderc.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "src/wrs/reference/sweeping_alias_table.hpp"

#include "merian/vk/memory/memory_allocator.hpp"
#include "src/wrs/algorithm/psa/PSA.hpp"
#include "src/wrs/algorithm/psa/construction/PSAC.hpp"
#include "src/wrs/algorithm/psa/sampling/SampleAliasTable.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/reference/psa_alias_table.hpp"
#include "src/wrs/reference/reduce.hpp"
#include <fmt/base.h>
#include <random>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

using namespace wrs;

void wrs::eval::write_psa_rmse_curves(const merian::ContextHandle& context) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);

    constexpr glsl::uint N = 1024 * 2048;
    constexpr uint64_t S = static_cast<uint64_t>(1e11);
    constexpr Distribution DIST = Distribution::PSEUDO_RANDOM_UNIFORM;
    const auto weights = wrs::generate_weights<float>(DIST, N);
    const float totalWeight = wrs::reference::kahan_reduction<float>(weights);

    // Construct alias table!
    wrs::PSAC psac{context, shaderCompiler};

    glsl::uint splitCount = (N + psac.getSplitSize() - 1) / psac.getSplitSize();

    wrs::PSAC::Buffers localPsac = wrs::PSACBuffers::allocate(
        alloc, N, 0, psac.getPrefixPartitionSize(), splitCount, merian::MemoryMappingType::NONE);
    wrs::PSAC::Buffers stagePsac =
        wrs::PSACBuffers::allocate(alloc, N, 0, psac.getPrefixPartitionSize(), splitCount,
                                   merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    constexpr bool USE_GPU_CONSTRUCTION = true;

    merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
    cmd->begin();
    if (USE_GPU_CONSTRUCTION) {
        {
            wrs::PSAC::Buffers::WeightsView stage{stagePsac.weights, N};
            wrs::PSAC::Buffers::WeightsView local{localPsac.weights, N};
            stage.upload<float>(weights);
            stage.copyTo(cmd, local);
            local.expectComputeRead(cmd);
        }
        psac.run(cmd, localPsac, N);

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     localPsac.aliasTable->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                          vk::AccessFlagBits::eShaderRead));
    } else {
        const auto aliasTable =
            wrs::reference::psa_alias_table<float, float, glsl::uint, std::allocator<float>>(
                weights, 2);

        wrs::PSAC::Buffers::AliasTableView stage{stagePsac.aliasTable, N};
        wrs::PSAC::Buffers::AliasTableView local{localPsac.aliasTable, N};
        void* mapped = stagePsac.aliasTable->get_memory()->map();
        std::memcpy(mapped, aliasTable.data(),
                    aliasTable.size() * sizeof(wrs::AliasTableEntry<float, glsl::uint>));
        stagePsac.aliasTable->get_memory()->unmap();
        stage.expectHostWrite();
        stage.copyTo(cmd, local);
        local.expectComputeRead(cmd);
    }

    cmd->end();
    queue->submit_wait(cmd);

    /* constexpr glsl::uint MAX_SAMPLING_STEP_SIZE = 0x3FFFFFFF; */
    constexpr glsl::uint MAX_SAMPLING_STEP_SIZE = 0x3FFFFFFF;
    constexpr glsl::uint SAMPLING_STEP_COUNT =
        (S + static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE) - 1) /
        static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE);
    constexpr glsl::uint RMSE_CURVE_TICKS = 100;
    constexpr glsl::uint SUBMIT_LIMIT = 4;

    wrs::SampleAliasTable sampleAlias{context, shaderCompiler};

    merian::BufferHandle samplesLocal = alloc->createBuffer(
        wrs::PSA::Buffers::SamplesLayout::size(MAX_SAMPLING_STEP_SIZE),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        merian::MemoryMappingType::NONE);

    merian::BufferHandle samplesStage = alloc->createBuffer(
        wrs::PSA::Buffers::SamplesLayout::size(MAX_SAMPLING_STEP_SIZE),
        vk::BufferUsageFlagBits::eTransferDst, merian::MemoryMappingType::HOST_ACCESS_RANDOM);

    constexpr bool USE_GPU_ACCELERATION = true;
    if (USE_GPU_ACCELERATION) {

        std::size_t s = S;
        std::mt19937 rng;
        std::uniform_int_distribution<glsl::uint> dist;

        wrs::eval::RMSECurveAcceleratedBuilder rmseCurveBuilder{
            context, shaderCompiler, localPsac.weights, totalWeight, N,
            wrs::eval::log10scale<uint64_t>(1000, S, RMSE_CURVE_TICKS)};
        std::vector<glsl::uint> samplesSection;

        for (std::size_t i = 0; i < SAMPLING_STEP_COUNT;) {

            merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
            cmd->begin();

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

                wrs::SampleAliasTableBuffers buffers;
                buffers.aliasTable = localPsac.aliasTable;
                buffers.samples = samplesLocal;

                glsl::uint seed = dist(rng);
                sampleAlias.run(cmd, buffers, N, s2, seed);

                rmseCurveBuilder.consume(cmd, samplesLocal, s2);
                ++i;
                ++x;
            }

            SPDLOG_INFO("Sectioned Sampling: {}/{} ~ {:.3}%", S - s, S,
                        100 * ((S - s) / static_cast<float>(S)));

            cmd->end();
            queue->submit_wait(cmd);
        }
        const auto& rmseCurve = rmseCurveBuilder.get();
        std::string path = "psa_rmse_curve.csv";
        SPDLOG_INFO("Writing RMSE-curve to \"{}\"", path);
        wrs::exp::CSVWriter<2> csv{{"sample_size", "RMSE"}, path};
        for (const auto& [s, rmse] : rmseCurve) {
            csv.pushRow(s, rmse);
        }
        SPDLOG_INFO("DONE");
    } else {

        std::size_t s = S;
        std::mt19937 rng;
        std::uniform_int_distribution<glsl::uint> dist;

        wrs::eval::RMSECurveSectionedBuilder<double, float, glsl::uint> rmseCurveBuilder{
            weights, wrs::eval::log10scale<uint64_t>(1000, S, RMSE_CURVE_TICKS)};
        std::vector<glsl::uint> samplesSection;

        for (std::size_t i = 0; i < SAMPLING_STEP_COUNT;) {

            merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
            cmd->begin();

            std::size_t s2 = s;
            if (s2 == 0) {
                continue;
            }
            if (s2 > MAX_SAMPLING_STEP_SIZE) {
                s2 = MAX_SAMPLING_STEP_SIZE;
            }
            s -= MAX_SAMPLING_STEP_SIZE;

            wrs::SampleAliasTableBuffers buffers;
            buffers.aliasTable = localPsac.aliasTable;
            buffers.samples = samplesLocal;

            glsl::uint seed = dist(rng);
            sampleAlias.run(cmd, buffers, N, s2, seed);

            wrs::SampleAliasTable::Buffers::SamplesView samplesStageView{samplesStage, s2};
            wrs::SampleAliasTable::Buffers::SamplesView samplesLocalView{samplesLocal, s2};
            samplesLocalView.expectComputeWrite();
            samplesLocalView.copyTo(cmd, samplesStageView);
            samplesStageView.expectHostRead(cmd);

            SPDLOG_INFO("Sectioned Sampling: {}/{} ~ {:.3}%", S - s, S,
                        100 * ((S - s) / static_cast<float>(S)));

            cmd->end();
            queue->submit_wait(cmd);

            auto samplesSection = samplesStageView.download<glsl::uint>();
            rmseCurveBuilder.consume(samplesSection);

            ++i;
        }
        const auto& rmseCurve = rmseCurveBuilder.get();
        std::string path = "psa_rmse_curve.csv";
        SPDLOG_INFO("Writing RMSE-curve to \"{}\"", path);
        wrs::exp::CSVWriter<2> csv{{"sample_size", "RMSE"}, path};
        for (const auto& [s, rmse] : rmseCurve) {
            csv.pushRow(s, rmse);
        }
        SPDLOG_INFO("DONE");
    }
}
