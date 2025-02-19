#pragma once

#include <concepts>
namespace device {

template <typename T>
concept block_wise_prefix_partition_compatible = std::same_as<float, T>;

}
