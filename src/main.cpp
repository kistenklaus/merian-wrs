#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "src/wrs/algorithm/mean/decoupled/test.hpp"
#include "src/wrs/algorithm/pack/scalar/test.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test.hpp"
#include "src/wrs/algorithm/psa/scalar/ScalarPsa.hpp"
#include "src/wrs/algorithm/split/scalar/test.hpp"
#include "src/wrs/eval/psa_ref_eval.hpp"
#include "src/wrs/eval/std_eval.hpp"
#include "src/wrs/eval/sweeping_eval.hpp"
#include "src/wrs/layout/BufferView.hpp"
#include "src/wrs/layout/StructLayout.hpp"
#include "src/wrs/test/test.hpp"
#include "wrs/algorithm/pack/simd/test.hpp"

#include <fmt/base.h>
#include <memory>
#include <set>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include "wrs/algorithm/psa/scalar/test.hpp"


int main() {
    spdlog::set_level(spdlog::level::debug);

    // Setup Vulkan context
    const auto core = std::make_shared<merian::ExtensionVkCore>(
        std::set<std::string>{"vk12/vulkanMemoryModel", "vk12/vulkanMemoryModelDeviceScope"});

    const auto debug_utils = std::make_shared<merian::ExtensionVkDebugUtils>(false);
    const auto resources = std::make_shared<merian::ExtensionResources>();
    const auto push_descriptor = std::make_shared<merian::ExtensionVkPushDescriptor>();
    const std::vector<std::shared_ptr<merian::Extension>> extensions = {
        core, resources, debug_utils, push_descriptor};
    const merian::ContextHandle context = merian::Context::create(extensions, "merian-example");

    if (!context) {
        throw std::runtime_error("Failed to create context!!!");
    }

    /*wrs::eval::write_sweeping_rmse_curves();*/
    /*wrs::eval::write_std_rmse_curves();*/
    /*wrs::eval::write_psa_rmse_curves();*/
    /* wrs::eval::write_std_rmse_curves(); */
    /* wrs::eval::write_psa_rmse_curves(); */
    //wrs::test::testTests();
    /* wrs::test::decoupled_mean::test(context); */
    //wrs::test::decoupled_prefix_partition::test(context);
     //wrs::test::scalar_split::test(context);
    //wrs::test::scalar_pack::test(context);
    wrs::test::simd_pack::test(context);
    //wrs::test::scalar_psa::test(context);
}


