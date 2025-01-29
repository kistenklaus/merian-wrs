#include "merian/vk/context.hpp"

#include "src/wrs/algorithm/prng/philox/test.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "src/wrs/algorithm/its/sampling/test.hpp"
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
#include "src/wrs/algorithm/prefix_sum/decoupled/test.hpp"
#include "src/wrs/algorithm/split/scalar/test.hpp"
#include "src/wrs/buffer_view/BufferView.hpp"
#include "src/wrs/eval/its_eval.hpp"
#include "src/wrs/eval/philox_eval.hpp"
#include "src/wrs/eval/psa_ref_eval.hpp"

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
#include "src/wrs/eval/psa_eval.hpp"

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

    wrs::foo();
    /* wrs::test::philox::test(context); */
    /* wrs::test::its_sampling::test(context); */


    /* wrs::eval::write_philox_rmse_curve(context); */
    /* wrs::eval::write_its_rmse_curves(context); */

    /* wrs::test::splitpack::test(context); */
    /* wrs::test::psac::test(context); */
    /* wrs::eval::write_psa_rmse_curves(context); */


    /* wrs::test::atomic_mean::test(context); */
}

