#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "src/wrs/algorithm/mean/decoupled/test.hpp"
#include "src/wrs/algorithm/prefix_partition/decoupled/test.hpp"
#include "src/wrs/algorithm/split/scalar/test.hpp"
#include "src/wrs/test/test.hpp"
#include <memory>
#include <set>
#include <spdlog/spdlog.h>
#include <stdexcept>

int main() {

    spdlog::set_level(spdlog::level::debug);

    // Setup Vulkan context.
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

    wrs::test::testTests();
    wrs::test::decoupled_mean::test(context);
    wrs::test::decoupled_prefix_partition::test(context);
    wrs::test::scalar_split::test(context);
}
