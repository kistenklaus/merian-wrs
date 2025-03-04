#pragma once

#include "src/host/reference/reduce.hpp"
#include "src/host/statistics/histogram.hpp"
#include <cmath>
#include <cstddef>
#include <span>
#include <spdlog/spdlog.h>
namespace host {

template <std::integral I, std::floating_point T>
T kl_divergence(std::span<const I> samples, std::span<const T> weights) {
    const T totalWeight = host::reference::reduce<T>(weights);

    const auto histogram = host::histogram(samples, weights.size());

    T sum = 0; // Running total sum
    T c = 0;   // Compensation term
    for (std::size_t i = 0; i < weights.size(); ++i) {
        T observed = histogram[i] / static_cast<T>(samples.size());
        T expected = weights[i] / totalWeight;

        if (observed > 0) {
            if (expected > 0) {
                T element = observed * std::log(observed / expected);

                T y = element - c; // Subtract compensation from current element
                T t = sum + y;     // Add the adjusted value to the sum
                c = (t - sum) - y; // Calculate the new compensation
                sum = t;           // Update the running total

            } else {
                SPDLOG_WARN("Distribution mismatch");
            }
        }
    }
    return sum;
}

} // namespace host
