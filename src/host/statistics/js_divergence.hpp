#pragma once

#include "src/host/reference/reduce.hpp"
#include "src/host/statistics/histogram.hpp"
#include <cassert>
#include <concepts>
#include <span>
#include <spdlog/spdlog.h>
namespace host {

template <std::floating_point T>
T js_weight_divergence(std::span<const T> observedWeights, std::span<const T> expectedWeights) {
    if (observedWeights.size() != expectedWeights.size()) {
        throw std::invalid_argument(
            "Observed and expected distributions must have the same number of bins.");
    }

    // Compute midpoint distribution
    std::vector<T> midpointWeights(observedWeights.size());

    for (std::size_t i = 0; i < observedWeights.size(); ++i) {
        midpointWeights[i] = (observedWeights[i] + expectedWeights[i]) / T(2.0);
    }

    // Compute JS divergence as the average of two KL divergences
    auto kl_divergence = [&](std::span<const T> P, std::span<const T> Q) -> T {
        assert(P.size() == Q.size());
        T sum = 0, c = 0; // Kahan summation for numerical stability
        for (std::size_t i = 0; i < P.size(); ++i) {
            if (P[i] > 0 && Q[i] > 0) { // Avoid log(0) issues
                T element = P[i]  * std::log(P[i] / Q[i]);
                T y = element - c;
                T t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
        }
        return sum / P.size();
    };

    T kl1 = kl_divergence(observedWeights, midpointWeights);
    T kl2 = kl_divergence(expectedWeights, midpointWeights);

    return (kl1 + kl2) / T(2.0);
}

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
