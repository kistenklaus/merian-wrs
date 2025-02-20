#pragma once

#include "src/host/reference/reduce.hpp"
#include "src/host/statistics/histogram.hpp"
#include <concepts>
#include <span>
#include <spdlog/spdlog.h>
namespace host {

template <std::integral I, std::floating_point T>
T js_divergence(std::span<const I> samples, std::span<const T> weights) {

    const T totalWeight = host::reference::reduce<T>(weights);

    const auto histogram = host::histogram(samples, weights.size());

    T sum = 0; // Running total sum
    T c = 0;   // Compensation term
    for (std::size_t i = 0; i < weights.size(); ++i) {
        T observed = histogram[i] / static_cast<T>(samples.size());
        T expected = weights[i] / totalWeight;
        T midpoint = (observed + expected) / T(2.0);

        if (observed > 0 && midpoint > 0) {
            T element = observed * std::log(observed / midpoint);

            T y = element - c; // Subtract compensation from current element
            T t = sum + y;     // Add the adjusted value to the sum
            c = (t - sum) - y; // Calculate the new compensation
            sum = t;           // Update the running total
        }
    }
    T kl1 = sum;

    sum = 0;
    c = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        T observed = histogram[i] / static_cast<T>(samples.size());
        T expected = weights[i] / totalWeight;
        T midpoint = (observed + expected) / T(2.0);

        if (expected > 0 && midpoint > 0) {
            T element = expected * std::log(expected / midpoint);

            T y = element - c; // Subtract compensation from current element
            T t = sum + y;     // Add the adjusted value to the sum
            c = (t - sum) - y; // Calculate the new compensation
            sum = t;           // Update the running total
        }
    }

    T kl2 = sum;


    return (kl1 + kl2) / T(2.0);
}
} // namespace host
