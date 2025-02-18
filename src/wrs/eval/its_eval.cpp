#include "./its_eval.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/shader/shader_compiler_shaderc.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "src/wrs/algorithm/its/sampling/InverseTransformSampling.hpp"
#include "src/wrs/algorithm/prefix_sum/decoupled/DecoupledPrefixSum.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/reference/reduce.hpp"

void wrs::eval::write_its_rmse_curves(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);
    merian::ShaderCompilerHandle shaderCompiler = std::make_shared<merian::SystemGlslcCompiler>(context);

    constexpr std::size_t N = 1024 * 2048;
    constexpr std::size_t S = 1e11;

    const std::vector<float> weights =
        wrs::generate_weights(Distribution::SEEDED_RANDOM_EXPONENTIAL, N);
    const float totalWeight = wrs::reference::kahan_reduction<float>(weights);

    wrs::InverseTransformSampling itsKernel{context, shaderCompiler};

    constexpr glsl::uint MAX_SAMPLING_STEP_SIZE = 0x3FFFFFFF;
    constexpr glsl::uint SAMPLING_STEP_COUNT =
        (S + static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE) - 1) /
        static_cast<uint64_t>(MAX_SAMPLING_STEP_SIZE);
    constexpr glsl::uint RMSE_CURVE_TICKS = 100;
    constexpr glsl::uint SUBMIT_LIMIT = 4;

    using SamplingBuffers = wrs::InverseTransformSampling::Buffers;
    SamplingBuffers stage = SamplingBuffers::allocate(
        alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, N, MAX_SAMPLING_STEP_SIZE);
    SamplingBuffers local =
        SamplingBuffers::allocate(alloc, merian::MemoryMappingType::NONE, N, MAX_SAMPLING_STEP_SIZE);

    wrs::DecoupledPrefixSum prefixSumKernel{context, shaderCompiler};
    using PrefixBuffers = wrs::DecoupledPrefixSum::Buffers;
    PrefixBuffers prefixLocal = PrefixBuffers::allocate(alloc, merian::MemoryMappingType::NONE, N,
                                                        prefixSumKernel.getPartitionSize());

    PrefixBuffers prefixStage =
        PrefixBuffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, N,
                                prefixSumKernel.getPartitionSize());
    {

        prefixLocal.prefixSum = local.cmf;
        prefixStage.prefixSum = stage.cmf;

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();
        PrefixBuffers::ElementsView stageView{prefixStage.elements, N};
        PrefixBuffers::ElementsView localView{prefixLocal.elements, N};
        stageView.upload<float>(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);

        prefixSumKernel.run(cmd, prefixLocal, N);

        PrefixBuffers::PrefixSumView prefixView{prefixLocal.prefixSum, N};
        prefixView.expectComputeRead(cmd);

        cmd->end();
        queue->submit_wait(cmd);
    }

    std::size_t s = S;
    std::mt19937 rng;
    std::uniform_int_distribution<glsl::uint> dist;

    wrs::eval::RMSECurveAcceleratedBuilder rmseCurveBuilder{
        context, shaderCompiler, prefixLocal.elements, totalWeight, N,
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
            s -= std::min<std::size_t>(s, MAX_SAMPLING_STEP_SIZE);

            
            glsl::uint seed = dist(rng);
            itsKernel.run(cmd, local, N, s2, seed);

            rmseCurveBuilder.consume(cmd, local.samples, s2);
            ++i;
            ++x;
        }

        SPDLOG_INFO("Sectioned Sampling: {}/{} ~ {:.3}%", S - s, S,
                    100 * ((S - s) / static_cast<float>(S)));

        cmd->end();
        queue->submit_wait(cmd);
    }

    const auto rmseCurve = rmseCurveBuilder.get();

    const std::string csvPath = "./its_rmse.csv";
    SPDLOG_INFO("Writing rmse results to \"{}\"",csvPath); 
    wrs::exp::CSVWriter<2> csv{{"sample_size", "Inverse-CMF"}, csvPath};

    for (auto rmse : rmseCurve) {
        csv.pushTupleRow(rmse);
    }
}
