#pragma once

#include <concepts>
namespace wrs {

template <typename T>
concept block_wise_prefix_partition_compatible = std::same_as<float, T>;

}
