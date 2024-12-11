#pragma once

#include <cmath>
namespace wrs::reference {

template <std::floating_point T>
T lanczos_gamma(T z) {
    static const T p[] = {
        676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7};
    static const T g = 7;

    if (z < 0.5) {
        // Reflection formula
        return M_PI / (std::sin(M_PI * z) * gamma(1 - z));
    }

    z -= 1;
    T x = 0.99999999999980993;
    for (size_t i = 0; i < sizeof(p) / sizeof(p[0]); ++i) {
        x += p[i] / (z + i + 1);
    }
    T t = z + g + 0.5;
    return std::sqrt(2 * M_PI) * std::pow(t, z + 0.5) * std::exp(-t) * x;
}

template <std::floating_point T>
T lanczos_incomplete_gamma(T s, T x) {
    T sum = 1 / s;
    T term = sum;
    for (int n = 1; n < 100; ++n) {
        term *= x / (s + n);
        sum += term;
        if (term < 1e-10) break; // Convergence
    }
    return sum * std::exp(-x) * std::pow(x, s);
}

}
