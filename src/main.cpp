#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/wrs/alias_table/baseline/algo/PartitionAndPrefixSum.hpp"
#include "src/wrs/alias_table/baseline/algo/PrefixSumAvg.hpp"
#include "wrs/alias_table/baseline/BaselineAliasTable.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <spdlog/spdlog.h>

constexpr bool RANDOM_WEIGHTS = true;

int main() {

    spdlog::set_level(spdlog::level::debug);

    // Setup Vulkan context.
    const auto core = std::make_shared<merian::ExtensionVkCore>(
        std::set<std::string>{"vk12/vulkanMemoryModel", "vk12/vulkanMemoryModelDeviceScope"});
    /* const auto vmm = std::make_shared<ExtensionVMM>(); */
    const auto debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
    const auto resources = std::make_shared<merian::ExtensionResources>();
    const auto push_descriptor = std::make_shared<merian::ExtensionVkPushDescriptor>();
    const std::vector<std::shared_ptr<merian::Extension>> extensions = {
        core, resources, debug_utils, push_descriptor};
    const merian::ContextHandle context = merian::Context::create(extensions, "merian-example");
    assert(context != nullptr);

    merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
    merian::QueueHandle queue = context->get_queue_GCT();
    merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
    merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
        std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
    query_pool->reset();
    profiler->set_query_pool(query_pool);

    /* constexpr size_t WEIGHT_COUNT = 1024 * 2048; */
    constexpr size_t WEIGHT_COUNT = 2048;
    /* constexpr size_t WEIGHT_COUNT = 128; */

    std::vector<float> weights(WEIGHT_COUNT, 1.0f);

    /* std::fill(weights.begin() + weights.size() / 2, weights.end(), 0.0f); */

    if constexpr (RANDOM_WEIGHTS) {
        std::mt19937 rng{0};
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = dist(rng);
        }
    }

    /* float sum = 0.0f; */
    /* for (size_t i = 0; i < weights.size(); ++i) { */
    /*   sum += weights[i]; */
    /* } */
    /* float avg = sum / weights.size(); */
    /*  */
    /* std::vector<float> prefix(128); */
    /* prefix[0] = weights[0]; */
    /* for (size_t i = 1; i < weights.size(); ++i) { */
    /*   prefix[i] = prefix[i - 1]; */
    /*   if (weights[i] > avg) { */
    /*     prefix[i] += weights[i]; */
    /*   } */
    /* } */
    /* for (size_t i = 0; i < weights.size(); ++i) { */
    /*   std::cout << "result[" << i << "] = " << prefix[i] << std::endl; */
    /* } */

    wrs::baseline::PartitionAndPrefixSum::testAndBench(context);
    return 0;

    wrs::baseline::BaselineAliasTable aliasTable(context, weights.size());

    merian::CommandPool cmd_pool(queue);
    vk::CommandBuffer cmd = cmd_pool.create_and_begin();
    aliasTable.set_weights(cmd, weights, profiler);

    aliasTable.build(cmd, profiler);
    auto [resultBuf, resultSize] = aliasTable.download_result(cmd, profiler);
    cmd.end();
    queue->submit_wait(cmd);

    uint8_t* result = resultBuf->get_memory()->map_as<uint8_t>();
    uint32_t splitInfoSize = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(float);
    for (size_t i = 0; i < resultSize; i += 1) {
        uint32_t heavyOffset = *reinterpret_cast<uint32_t*>(result + splitInfoSize * i);
        uint32_t lightOffset =
            *reinterpret_cast<uint32_t*>(result + splitInfoSize * i + sizeof(uint32_t));
        float spill = *reinterpret_cast<float*>(result + splitInfoSize * i + sizeof(uint32_t) +
                                                   sizeof(uint32_t));
        /* std::cout << "{" << heavyOffset << "," << lightOffset << "," << spill << "}" << std::endl; */

        /* std::cout << "result[" << i << "] = " << result[i] << '\n'; */
    }
    /* std::cout << "heavyCount = " << heavyCount << std::endl; */
    resultBuf->get_memory()->unmap();
    /* std::cout << "GPU-ReduceResult: " << result << std::endl; */

    profiler->collect(true);
    std::cout << merian::Profiler::get_report_str(profiler->get_report()) << std::endl;

    aliasTable.cpuValidation(queue, cmd_pool, weights.size());
}
