#include "merian/vk/context.hpp"
#include "src/bench/sample_throughput2.hpp"
#include "src/device/partition/test.hpp"

#include "src/bench/wrs_rmse_sweep.hpp"
#include "src/bench/wrs.hpp"
#include "src/bench/wrs_rmse.hpp"
#include "src/device/prefix_sum/test.hpp"
#include "src/bench/psa_inline_splitpack_work.hpp"
#include "src/bench/scan.hpp"
#include "src/bench/psa_inline_splitpack_all.hpp"
#include "src/bench/psa_splitpack.hpp"
#include "src/bench/psa_inline_splitpack_splitSize.hpp"
#include "src/bench/psa_pack_throughput.hpp"
#include "src/bench/cutpoint_latency.hpp"
#include "src/bench/sample_throughput.hpp"
#include "src/bench/psa_split.hpp"
#include "src/bench/psa_pack.hpp"
#include "src/bench/psa_pack_work.hpp"
#include "src/bench/psa_split2.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_float_atomics.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "src/device/prefix_partition/test.hpp"
#include "src/device/wrs/alias/psa/test.hpp"
#include "src/device/wrs/alias/psa_plus/greedy/test.hpp"
#include "src/device/wrs/test.hpp"
#include <dlfcn.h>
#include <fmt/base.h>
#include <memory>
#include <set>
#include <spdlog/spdlog.h>
#include <stdexcept>

int main() {

    spdlog::set_level(spdlog::level::debug);

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
        extensions, "merian-example", VK_MAKE_VERSION(1, 0, 0), 1, VK_API_VERSION_1_3, false, -1,
        // AMD Radeon Graphics
        /* 5710,  */
        // NVIDIA RTX 4070
        10118,
        //
        ""
        );

    if (!context) {
        throw std::runtime_error("Failed to create context!!!");
    }

    /* device::wrs_rmse_sweep::benchmark(context); */

    /* device::wrs_rmse::benchmark(context); */
    /* device::wrs::benchmark(context); */

    /* host::test::testTests(); */

    /* device::wrs_rmse::benchmark(context); */
    /* device::sample_throughput2::benchmark(context); */

    /* device::test::mean::test(context); */
    /* device::test::partition::test(context); */
    /* device::test::prefix_sum::test(context); */
    /* device::test::prefix_partition::test(context); */

    /* device::test::psa_greedy::test(context); */
    device::test::wrs::test(context);

    /* device::wrs::benchmark(context); */
    /* device::scan::benchmark(context); */
    /* device::block_scan::benchmark(context); */
    /* device::partition_scan::benchmark(context); */

    /* device::memcpy::benchmark(context); */

    /* device::test::psa::test(context); */

    /* device::sample_throughput::benchmark(context); */
    /* device::cutpoint_latency::benchmark(context); */
    /* device::psa_split::benchmark(context); */
    /* device::psa_split2::benchmark(context); */
    /* device::psa_pack::benchmark(context); */
    /* device::psa_pack_throughput::benchmark(context); */
    /* device::psa_splitpack::benchmark(context); */
    /* device::psa_inline_splitpack_work::benchmark(context); */
    /* device::psa_inline_splitpack_all::benchmark(context); */

    /* device::psa_pack_work::benchmark(context); */

    /* device::psa_pack_throughput::benchmark(context); */

    /* device::psa_inline_splitpack_all::benchmark(context); */

}
