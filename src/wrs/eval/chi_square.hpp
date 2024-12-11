#pragma once

#include "src/wrs/eval/histogram.hpp"
#include "src/wrs/reference/gamma.hpp"
#include "src/wrs/reference/reduce.hpp"
#include <cassert>
#include <cmath>
#include <concepts>
#include <fmt/base.h>
#include <limits>
#include <span>

namespace wrs::eval {

// Compute critical value for Chi-Square distribution
// - E: Error type (must satisfy std::floating_point)
template <std::floating_point E> E chi_square_critical_value(std::size_t populationSize, E alpha) {
    assert(populationSize > 1);
    std::size_t degreesOfFreedom = populationSize - 1;
    assert(degreesOfFreedom > 0);
    assert(alpha > 0 && alpha < 1);

    // Approximation of the inverse CDF for Chi-Square distribution using Wilson-Hilferty
    // transformation Source: Numerical Recipes in C (CHAT-GPT!!!)
    E z = -std::log(alpha);
    E t = std::sqrt(2 * degreesOfFreedom) * (z - 1.0 + (2.0 / (9.0 * degreesOfFreedom)));
    return degreesOfFreedom * std::pow((1.0 + t / std::sqrt(2.0 * degreesOfFreedom)), 3.0);
}

template <std::floating_point E, std::integral I>
E chi_square(std::span<const I> samples, std::span<const E> weights) {
    assert(!samples.empty());
    assert(!weights.empty());

    auto observed = wrs::eval::histogram<I>(samples, weights.size());

    E totalWeight = wrs::reference::kahan_reduction<E>(weights);
    assert(totalWeight > 0);

    E chiSquare = E(0);
    for (size_t i = 0; i < observed.size(); ++i) {
        assert(weights[i] > 0);
        E expectedValue = (weights[i] / totalWeight) * static_cast<E>(samples.size());
        assert(expectedValue > 0); // Expected values must be positive
        chiSquare += std::pow(static_cast<E>(observed[i]) - expectedValue, 2) / expectedValue;
    }
    return chiSquare;
}

namespace chi2_internal {

template <std::floating_point E> E cdf(E x, std::size_t degreesOfFreedom) {
    assert(degreesOfFreedom > 0);
    E k = degreesOfFreedom / 2.0;
    E chi = x / 2.0;
    auto incomp_gamma = wrs::reference::lanczos_incomplete_gamma<E>(k, chi);
    auto gamma = wrs::reference::lanczos_gamma<E>(k);
    return incomp_gamma / gamma;
}

template <std::floating_point T> T lanczos_gamma(T z) {
    static const T p[] = {676.5203681218851,     -1259.1392167224028,  771.32342877765313,
                          -176.61502916214059,   12.507343278686905,   -0.13857109526572012,
                          9.9843695780195716e-6, 1.5056327351493116e-7};
    static const T g = 7;

    if (z < 0.5) {
        // Reflection formula
        return M_PI / (std::sin(M_PI * z) * wrs::eval::chi2_internal::lanczos_gamma(1 - z));
    }

    z -= 1;
    T x = 0.99999999999980993;
    for (size_t i = 0; i < sizeof(p) / sizeof(p[0]); ++i) {
        x += p[i] / (z + i + 1);
    }
    T t = z + g + 0.5;
    return std::sqrt(2 * M_PI) * std::pow(t, z + 0.5) * std::exp(-t) * x;
}

template <std::floating_point T> T lanczos_incomplete_gamma(T s, T x) {
    T sum = 1 / s;
    T term = sum;
    for (int n = 1; n < 100; ++n) {
        term *= x / (s + n);
        sum += term;
        if (std::abs(term) < std::numeric_limits<T>::epsilon() * 10)
            break; // Convergence
    }
    return sum * std::exp(-x) * std::pow(x, s);
}

} // namespace __chi_squared

template <std::floating_point E> E chi_square_alpha(E chiSquare, std::size_t populationSize) {
    assert(populationSize > 1 && "Population size must be greater than 1");
    assert(chiSquare > 0);
    std::size_t degreesOfFreedom = populationSize - 1;
    return 1.0 - chi2_internal::cdf<E>(chiSquare, degreesOfFreedom);
}

} // namespace wrs::eval
