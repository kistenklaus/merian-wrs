#include "./its_eval.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "src/wrs/algorithm/its/sampling/InverseTransformSampling.hpp"
#include "src/wrs/eval/logscale.hpp"
#include "src/wrs/eval/rms.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/export/csv.hpp"
#include "src/wrs/why.hpp"

void wrs::eval::write_its_rmse_curves(const merian::ContextHandle& context) {

    auto resources = context->get_extension<merian::ExtensionResources>();
    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

    std::size_t N = 1024 * 2048;
    std::size_t S = 10e7;

    std::vector<float> weights = wrs::generate_weights(Distribution::SEEDED_RANDOM_UNIFORM, N);
    std::vector<float> cmf = wrs::reference::prefix_sum<float>(weights);

    wrs::InverseTransformSampling itsKernel{context, 512};

    using Buffers = wrs::InverseTransformSampling::Buffers;
    Buffers stage =
        Buffers::allocate(alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, cmf.size(), S);
    Buffers local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, cmf.size(), S);

    Buffers::CMFView cmfStage{stage.cmf, cmf.size()};
    Buffers::CMFView cmfLocal{local.cmf, cmf.size()};

    Buffers::SamplesView samplesStage{stage.samples, S};
    Buffers::SamplesView samplesLocal{local.samples, S};

    vk::CommandBuffer cmd = cmdPool->create_and_begin();

    cmfStage.upload<float>(cmf);
    cmfStage.copyTo(cmd, cmfLocal);
    cmfLocal.expectComputeRead(cmd);

    itsKernel.run(cmd, local, N, S);

    samplesLocal.copyTo(cmd, samplesStage);
    samplesStage.expectHostRead(cmd);

    cmd.end();
    queue->submit_wait(cmd);

    std::vector<wrs::glsl::uint> samples = samplesStage.download<wrs::glsl::uint>();

    std::size_t ticks = 100;
    wrs::eval::log10::IntLogScaleRange<glsl::uint> scale =
        wrs::eval::log10scale<wrs::glsl::uint>(1, S, ticks);

    auto rmseCurve = wrs::eval::rmse_curve<float, glsl::uint>(weights, samples, scale, std::nullopt);
  

    const std::string csvPath = "./its_rmse.csv";
    SPDLOG_INFO(fmt::format("Combining computed RSME curves and writing results to {}", csvPath));

    wrs::exp::CSVWriter<2> csv{{"sample_size", "Inverse-CMF"}, csvPath};
    
    for (auto rmse : rmseCurve) {
      csv.pushTupleRow(rmse);
    }
}
