#include "./hst_eval.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler_shaderc.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "src/wrs/algorithm/hs/HS.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/reference/reduce.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <random>

using namespace wrs;

constexpr glsl::uint HSTC_WORKGROUP_SIZE = 512;
constexpr glsl::uint SVO_WORKGROUP_SIZE = 512;
constexpr glsl::uint SAMPLING_WORKGROUP_SIZE = 512;
constexpr glsl::uint EXPLODE_WORKGROUP_SIZE = 512;
constexpr glsl::uint EXPLODE_ROWS = 8;
constexpr glsl::uint EXPLODE_LOOKBACK_DEPTH = 32;

constexpr Distribution DISTRIBUTION = wrs::Distribution::PSEUDO_RANDOM_UNIFORM;
constexpr glsl::uint N = 1024 * 2048;
constexpr glsl::uint S = 1e8;

constexpr glsl::uint RMSE_TICKS = 10;

float compute_rmse(const wrs::HS& hs,
                   const merian::CommandPoolHandle& cmdPool,
                   const merian::QueueHandle& queue,
                   const wrs::HS::Buffers& local,
                   const wrs::HS::Buffers& stage,
                   glsl::uint S,
                   std::span<const float> weights,
                   float totalWeight) {

    wrs::hst::HSTRepr repr{N};

    merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
    cmd->begin();

    // Upload sample count
    glsl::uint* mapped = stage.outputSensitiveSamples->get_memory()->map_as<glsl::uint>();
    mapped[repr.size()] = S;
    stage.outputSensitiveSamples->get_memory()->unmap();
    vk::BufferCopy copy{
        repr.size() * sizeof(glsl::uint),
        repr.size() * sizeof(glsl::uint),
        sizeof(glsl::uint),
    };
    cmd->copy(stage.outputSensitiveSamples, local.outputSensitiveSamples, copy);

    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                 local.outputSensitiveSamples->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                              vk::AccessFlagBits::eShaderRead));

    hs.run(cmd, local, N, S);

    wrs::HS::Buffers::SamplesView stageView{stage.samples, S};
    wrs::HS::Buffers::SamplesView localView{local.samples, S};
    localView.expectComputeWrite();
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);

    cmd->end();
    queue->submit_wait(cmd);

    std::vector<glsl::uint> samples = stageView.download<glsl::uint>();
    /* std::shuffle(samples.begin(), samples.end(), std::random_device{}); */
    assert(samples.size() == S);

    float rmse = wrs::eval::rmse<float, glsl::uint>(weights, samples, totalWeight);

    fmt::println("S:{}  => RMSE:{}", S, rmse);
    return rmse;
}

void wrs::eval::write_hst_rmse_curves(const merian::ContextHandle& context) {
    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);
    merian::ShaderCompilerHandle shaderCompiler =
        std::make_shared<merian::SystemGlslcCompiler>(context);

    wrs::HS::Buffers local = wrs::HS::Buffers::allocate(alloc, merian::MemoryMappingType::NONE, N,
                                                        S, EXPLODE_WORKGROUP_SIZE * EXPLODE_ROWS);
    wrs::HS::Buffers stage =
        wrs::HS::Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, N, S,
                                   EXPLODE_WORKGROUP_SIZE * EXPLODE_ROWS);

    std::vector<float> weights = wrs::generate_weights(DISTRIBUTION, N);
    float totalWeight = wrs::reference::kahan_reduction<float>(weights);

    merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
    cmd->begin();

    wrs::HS::Buffers::WeightTreeView stageView{stage.weightTree, N};
    wrs::HS::Buffers::WeightTreeView localView{local.weightTree, N};
    stageView.upload<float>(weights);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);

    cmd->end();
    queue->submit_wait(cmd);

    wrs::HS hs{context,
               shaderCompiler,
               HSTC_WORKGROUP_SIZE,
               SVO_WORKGROUP_SIZE,
               SAMPLING_WORKGROUP_SIZE,
               EXPLODE_WORKGROUP_SIZE,
               EXPLODE_ROWS,
               EXPLODE_LOOKBACK_DEPTH};

    wrs::exp::CSVWriter<2> rmseCurve{{"sample_size", "RMSE"}, "hst_rmse.csv"};

    for (const auto& s : wrs::eval::log10scale<glsl::uint>(1000, S, RMSE_TICKS)) {
        float rmse = compute_rmse(hs, cmdPool, queue, local, stage, s, weights, totalWeight);
        rmseCurve.pushRow(s, rmse);
    }
}
