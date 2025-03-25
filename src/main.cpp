#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_float_atomics.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "src/bench/block_scan.hpp"
#include "src/bench/cutpoint_latency.hpp"
#include "src/bench/memcpy.hpp"
#include "src/bench/prefix_partition.hpp"
#include "src/bench/psa_split.hpp"
#include "src/bench/psa_split_latency.hpp"
#include "src/bench/psa_splitpack_sweep.hpp"
#include "src/bench/scan.hpp"
#include "src/bench/scan_error.hpp"
#include "src/bench/wrs_rmse.hpp"
#include "src/device/mean/test.hpp"
#include "src/device/partition/test.hpp"
#include "src/device/prefix_partition/test.hpp"
#include "src/device/prefix_sum/test.hpp"
#include "src/device/wrs/alias/psa/test.hpp"
#include "src/device/wrs/test.hpp"
#include "src/host/test/context.hpp"
#include <dlfcn.h>
#include <fmt/base.h>
#include <memory>
#include <set>
#include <spdlog/spdlog.h>
#include <stdexcept>

merian::ContextHandle createContext();

int main() {

    // Setup merian. (e.g. VkInstance, VkPhysicalDevice, VkDevice, .....)
    const auto context = createContext();
    /*
     * The test suit verifies all invariants of algorithms.
     * Verification is mostly done on the CPU, therefor some tests might take a while.
     * IMPORTANTLY, tests can fail with meaning for example due to numerical instability,
     * that only occurs after a certain input size.
     * ERRORs generally indicate something, went wrong, some WARNINGs are expected.
     *
     * If you are only interessted in some performance measurements, just enable all
     * tests we print a pretty log message at the end,
     * which prints latencies with a couple of input sizes.
     *
     * For more indepth measurements use enable ENABLE_BENCHMARKS, this
     * will perform large benchmarks and write the results in the export/ directory.
     * We attach some python scripts for easy plotting of the results.
     */
    constexpr bool ENABLE_TESTS = true;
    constexpr bool ENABLE_BENCHMARKS = false;

    if (ENABLE_TESTS) {
        const auto testContext = host::test::setupTestContext(context);
        /*
         * Feel free to comment some tests if you are only interessted in
         * a certain algorithm.
         *
         * The complete test suit takes around 1 to 5 minutes! (RTX 4070)
         */
        device::mean::test(testContext); // requires floating point atomics
        device::partition::test(testContext);
        device::prefix_partition::test(testContext);
        device::prefix_sum::test(testContext);
        device::psa::test(testContext);
        device::wrs::test(testContext);

        testContext.printPrettyLog();
    }

    if (ENABLE_BENCHMARKS) {
        /*
         * All benchmarks, which are shown in the paper.
         * We configure all benchmarks by default to have slightly lower resolution that what we
         * show in the paper.
         *
         * IMPORTANT: If you actually wanna run benchmarks make sure that you lock the memory clock,
         * most GPUs can easily work with a constant memory clock speed without overheating.
         * If you know what your doing, locking the gpu clock can also help with reducing
         * variance, but it might damage your device if you lock both as the GPU can no longer
         * throttle!!!!
         *
         * Running benchmarks still takes very long, we configures
         * all benchmarks to by slightly less inaccurate (i.e. less iterations and less ticks),
         * however expect benchmarks to take a while.
         * On a RTX 4070, with a Ryzen 7 7800x3d it takes 2-3 days to run the
         * complete suite.
         *
         * NOTE: Anything ending with sweep takes a eternity because it
         * sweeps a lot of different shader configurations which requires constructing
         * a new pipelines, which takes ages.
         * The sweeps are not reconfigured to run faster as they only really make sense
         * when we sweep large.
         */

        // =========== simple and fast ==============
        device::memcpy::benchmark(context);
        device::block_scan::benchmark(context);
        device::scan::benchmark(context);
        device::partition_scan::benchmark(context);
        device::cutpoint_latency::benchmark(context);
        device::psa_split::benchmark(context);
        device::psa_split_latency::benchmark(context);
        // TODO sample throughput
        // TODO sample throughput2

        // =========== error metrics =================
        device::scan_error::benchmark(context);
        device::wrs_rmse::benchmark(context);

        // Sweeps take a eternity, either run them offline somewhere or over night.
        device::psa_splitpack_sweep::benchmark(context);

        // The final Figure of the paper comes from these two sweeps
        // TODO wrs.cpp
        // TODO wrs_rmse_sweep.cpp
    }
}

/// Just a helper function, which does the initalization
merian::ContextHandle createContext() {
    // Setup logging
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    // Setup Vulkan context
    const auto core = std::make_shared<merian::ExtensionVkCore>(
        std::set<std::string>{"vk12/vulkanMemoryModel", "vk12/vulkanMemoryModelDeviceScope",
                              "vk12/shaderBufferInt64Atomics"});

    const auto floatAtomics =
        std::make_shared<merian::ExtensionVkFloatAtomics>(std::set<std::string>{
            "shaderBufferFloat32Atomics",
            "shaderBufferFloat32AtomicAdd",
        });

    const auto debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(true);
    const auto resources = std::make_shared<merian::ExtensionResources>();
    const auto push_descriptor = std::make_shared<merian::ExtensionVkPushDescriptor>();
    const std::vector<std::shared_ptr<merian::Extension>> extensions = {
        core, floatAtomics, resources, debug_utils, push_descriptor};

    const merian::ContextHandle context = merian::Context::create(
        extensions, "merian-alias-table", VK_MAKE_VERSION(1, 0, 0), 1, VK_API_VERSION_1_3, false);

    if (!context) {
        throw std::runtime_error("Failed to create context!!!");
    }
    return context;
}
