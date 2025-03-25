#include "./wrs.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/device/prefix_sum/PrefixSum.hpp"
#include "src/device/prefix_sum/block_scan/BlockScan.hpp"
#include "src/device/prefix_sum/block_scan/BlockScanVariant.hpp"
#include "src/host/export/csv.hpp"
#include "src/host/export/logscale.hpp"
#include "src/host/gen/weight_generator.h"
#include "src/host/reference/prefix_sum.hpp"
#include "src/host/reference/reduce.hpp"
#include "vulkan/vulkan_handles.hpp"
#include <algorithm>
#include <csignal>
#include <fmt/base.h>
#include <spdlog/spdlog.h>

namespace device::scan_error {

enum Method {
    Sequential,
    SequentialKahan,
    SingleDispatch,
};

Method methods[]{
    Sequential,
    SequentialKahan,
    SingleDispatch,
};

std::size_t min_N = (1 << 16);
std::size_t max_N = (1 << 28);
std::size_t ticks = 100;

void errorOfMethod(Method method,
                   std::span<const float> weights,
                   std::span<const long double> ref,
                   host::exp::CSVWriter<6>& csv,
                   const merian::ContextHandle& context) {
    switch (method) {
    case Sequential: {
        SPDLOG_INFO("Compute error of SequentialScan (CPU)");
        for (std::size_t n : host::exp::log10scale(min_N, max_N, ticks)) {
            std::span subset{weights.begin(), weights.begin() + n};
            const auto scan = host::reference::sequential_prefix_sum<float>(subset);
            float abs_error = 0;
            float abs_error2 = 0;
            for (std::size_t i = 1; i < n; ++i) {
                float abs_diff = std::abs(scan[i] - ref[i]);
                abs_error += abs_diff;

                float reconstructed_weight = scan[i] - scan[i - 1];
                float abs_diff2 = std::abs(reconstructed_weight - weights[i]);
                abs_error2 += abs_diff2;
            }
            float totalWeight = host::reference::reduce<float>(subset);
            float rel_error = abs_error / totalWeight;
            float rel_error2 = abs_error2 / totalWeight;
            csv.pushRow(n, "sequential-scan", abs_error, rel_error, abs_error2, rel_error2);
        }
        break;
    }
    case SequentialKahan: {
        SPDLOG_INFO("Compute error of SequentialScan-Kahan (CPU)");
        for (std::size_t n : host::exp::log10scale(min_N, max_N, ticks)) {
            std::span subset{weights.begin(), weights.begin() + n};
            const auto scan = host::reference::prefix_sum<float>(subset);
            float abs_error = 0;
            float abs_error2 = 0;
            for (std::size_t i = 1; i < n; ++i) {
                float abs_diff = std::abs(scan[i] - ref[i]);
                abs_error += abs_diff;

                float reconstructed_weight = (scan[i] - scan[i - 1]);
                float abs_diff2 = std::abs(reconstructed_weight - subset[i]);
                abs_error2 += abs_diff2;
            }
            float totalWeight = host::reference::reduce<float>(subset);
            float rel_error = abs_error / totalWeight;
            float rel_error2 = abs_error2 / totalWeight;
            csv.pushRow(n, "sequential-scan-kahan", abs_error, rel_error, abs_error2, rel_error2);
        }
        break;
    }
    case SingleDispatch: {
        SPDLOG_INFO("Compute error of Single-dispatch (GPU)");
        merian::QueueHandle queue = context->get_queue_GCT();
        merian::ShaderCompilerHandle shaderCompiler =
            std::make_shared<merian::SystemGlslcCompiler>(context);
        merian::CommandPoolHandle cmdPool = std::make_shared<merian::CommandPool>(queue);

        using Scan = device::PrefixSum<float>;
        using Config = Scan::Config;
        using Buffers = Scan::Buffers;
        const Config config = DecoupledPrefixSumConfig();

        Scan scanKernel{context, shaderCompiler, config};

        assert(scanKernel.maxElementCount() >= max_N);

        const auto& resourceExt = context->get_extension<merian::ExtensionResources>();
        assert(resourceExt != nullptr);
        auto alloc = resourceExt->resource_allocator();
        Buffers stage = Buffers::allocate(
            alloc, merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE, config, max_N);
        Buffers local = Buffers::allocate(alloc, merian::MemoryMappingType::NONE, config, max_N);

        Buffers::ElementsView<float> stageView{stage.elements, max_N};
        Buffers::ElementsView<float> localView{local.elements, max_N};
        Buffers::PrefixSumView<float> resStage{stage.prefixSum, max_N};
        Buffers::PrefixSumView<float> resLocal{local.prefixSum, max_N};

        merian::CommandBufferHandle cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
        cmd->begin();
        stageView.upload(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);

        scanKernel.run(cmd, local, max_N);

        resLocal.expectComputeWrite();
        resLocal.copyTo(cmd, resStage);
        resStage.expectHostRead(cmd);
        cmd->end();
        queue->submit_wait(cmd);

        auto scan = resStage.download<float>();

        for (std::size_t n : host::exp::log10scale(min_N, max_N, ticks)) {
            std::span subset{weights.begin(), weights.begin() + n};
            std::span subscan{scan.begin(), scan.end() + n};
            float abs_error = 0;
            float abs_error2 = 0;
            for (std::size_t i = 1; i < n; ++i) {
                float abs_diff = std::abs(subscan[i] - ref[i]);
                abs_error += abs_diff;

                float reconstructed_weight = scan[i] - scan[i - 1];
                float abs_diff2 = std::abs(reconstructed_weight - weights[i]);
                abs_error2 += abs_diff2;
            }
            float totalWeight = host::reference::reduce<float>(subset);
            float rel_error = abs_error / totalWeight;
            float rel_error2 = abs_error2 / totalWeight;
            csv.pushRow(n, "single-dispatch", abs_error, rel_error, abs_error2, rel_error2);
        }

        break;
    }
    }
}

void benchmark(const merian::ContextHandle& context) {
    // Setup vulkan resources
    SPDLOG_INFO("Benchmarking scan error");

    std::string path = "export/scan/error.csv";
    host::exp::CSVWriter<6> csv(
        {"N", "method", "abs_error", "rel_error", "abs_error2", "rel_error2"}, path);

    auto weights = host::generate_weights<float>(host::Distribution::PSEUDO_RANDOM_UNIFORM, max_N);

    auto reference = host::reference::prefix_sum<long double>(weights);

    for (const auto& method : methods) {
        errorOfMethod(method, weights, reference, csv, context);
    }

    SPDLOG_INFO("Writing results to {}", path);
}

} // namespace device::scan_error
