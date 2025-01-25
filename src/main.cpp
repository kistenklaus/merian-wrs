#include "merian/vk/context.hpp"

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_float_atomics.hpp"
#include "src/wrs/algorithm/hs/HSTRepr.hpp"
#include "src/wrs/algorithm/hs/explode/test.hpp"
#include "src/wrs/algorithm/hs/hstc/test.hpp"
#include "src/wrs/algorithm/hs/sampling/test.hpp"
#include "src/wrs/algorithm/hs/svo/test.hpp"
#include "src/wrs/algorithm/its/test.hpp"
#include "src/wrs/algorithm/mean/atomic/test.hpp"
#include "src/wrs/algorithm/mean/decoupled/test.hpp"
#include "src/wrs/algorithm/pack/scalar/test.hpp"
#include "src/wrs/algorithm/pack/simd/test.hpp"
#include "src/wrs/algorithm/prefix_sum/decoupled/test.hpp"
#include "src/wrs/algorithm/split/scalar/test.hpp"

#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "src/wrs/algorithm/hs/test.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test.hpp"
#include "src/wrs/algorithm/psa/construction/test.hpp"
#include "src/wrs/algorithm/psa/test.hpp"
#include "src/wrs/algorithm/splitpack/test.hpp"
#include "src/wrs/bench/hst.hpp"
#include "src/wrs/bench/its.hpp"
#include "src/wrs/bench/psa.hpp"
#include "src/wrs/eval/hst_eval.hpp"
#include "src/wrs/eval/hst_std_eval.hpp"

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
        std::set<std::string>{"vk12/vulkanMemoryModel", "vk12/vulkanMemoryModelDeviceScope"});

    const auto floatAtomics =
        std::make_shared<merian::ExtensionVkFloatAtomics>(std::set<std::string>{
            "shaderBufferFloat32Atomics",
            "shaderBufferFloat32AtomicAdd",
        });

    const auto debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
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
        "");

    if (!context) {
        throw std::runtime_error("Failed to create context!!!");
    }

    wrs::test::atomic_mean::test(context);

    /* wrs::test::its::test(context); */
    /* wrs::test::splitpack::test(context); */

    /* wrs::test::hstc::test(context); */
    /* wrs::test::hs_svo::test(context); */
    /* wrs::test::hst_sampling::test(context); */
    /* wrs::test::hs_explode::test(context); */
    /* wrs::test::hs::test(context); */

    /* wrs::eval::write_hst_rmse_curves(context); */
    /* wrs::eval::write_hst_std_rmse_curves(); */

    /* wrs::bench::hst::write_bench_results(context); */

    /* wrs::test::decoupled_prefix_partition::test(context); */

    /* wrs::bench::its::write_bench_results(context); */

    /* wrs::test::psa::test(context); */

    /* wrs::bench::psa::write_bench_results(context); */

    /* wrs::test::psac::test(context); */

    // wrs::eval::write_its_rmse_curves(context);

    /*wrs::eval::write_sweeping_rmse_curves();*/
    /*wrs::eval::write_std_rmse_curves();*/
    /* wrs::eval::write_psa_rmse_curves(context); */
    /* wrs::eval::write_std_rmse_curves(); */
    /* wrs::eval::write_psa_rmse_curves(); */
    // wrs::test::testTests();
    /* wrs::test::decoupled_mean::test(context);  */
    //

    /* wrs::test::scalar_split::test(context); */

    /* wrs::test::scalar_pack::test(context); */

    /* wrs::test::simd_pack::test(context); */
    /* wrs::test::philox::test(context); */
    /* wrs::test::its::test(context); */
    /* wrs::test::sample_alias::test(context); */

    /* wrs::test::decoupled_prefix_sum::test(context); */
}
