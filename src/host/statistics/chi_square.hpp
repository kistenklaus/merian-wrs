#pragma once

#include "src/host/reference/reduce.hpp"
#include "src/host/statistics/histogram.hpp"
#include <cmath>
#include <span>
namespace host {

double chi_square_p_value(double chi2, int df);

inline double chi_square_z_score(double chi2, std::size_t df) {
  double chiMean = static_cast<double>(df);
  double chiVar = std::sqrt(2.0 * static_cast<double>(df));
  return (chi2 - chiMean) / chiVar;
}

template <std::integral T, std::floating_point W>
W chi_square(std::span<const T> samples, std::span<const W> weights) {
    const W totalWeight = host::reference::reduce<W>(weights);

    const auto histogram = host::histogram<T>(samples, weights.size());

    W sum = 0; // Running total sum
    W c = 0;   // Compensation term
    for (std::size_t i = 0; i < weights.size(); ++i) {
        W o = static_cast<W>(histogram[i]);
        W e = (weights[i] * static_cast<W>(samples.size())) / totalWeight;
        W element = ((o - e) * (o - e)) / e;
        
        W y = element - c; // Subtract compensation from current element
        W t = sum + y;     // Add the adjusted value to the sum
        c = (t - sum) - y; // Calculate the new compensation
        sum = t;           // Update the running total
    }

    return sum;
}

} // namespace host
