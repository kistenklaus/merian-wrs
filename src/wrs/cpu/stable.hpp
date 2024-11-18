#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fmt/base.h>
#include <fmt/format.h>
#include <iostream>
#include <numeric>
#include <spdlog/spdlog.h>
#include <tuple>
#include <vector>

namespace wrs::cpu::stable {

template <typename T> static std::vector<T> prefix_sum(const std::vector<T>& weights) {
    std::vector<T> prefix = weights;
    uint64_t N = weights.size();
    for (uint64_t shift = 1; shift <= N; shift <<= 1) {
        for (size_t i = N - 1; i >= shift; --i) {
            prefix.at(i) += prefix.at(i - shift);
        }
    }
    return prefix;
}

template <typename T> static std::vector<T> work_efficient_prefix_sum(const std::vector<T>& weights) {
  // up sweep 
  std::vector<T> prefix = weights;
  uint64_t N = weights.size();
  for (size_t d = 0; d < 0; d++) {
    /* kkfor (size_t p =  */
  }
}

template <typename T>
static std::tuple<std::vector<T>, std::vector<T>> partition(const std::vector<T>& weights,
                                                            T pivot) {
    std::vector<T> heavy;
    std::vector<T> light;
    for (auto w : weights) {
        if (w > pivot) {
            heavy.push_back(w);
        } else {
            light.push_back(w);
        }
    }
    return std::make_tuple(std::move(heavy), std::move(light));
}

struct SplitDescriptor {
    int heavyCount;
    int lightCount;
    float spill;
    float sigma;
};

template <typename T>
static SplitDescriptor split(const std::vector<T>& heavy,
                             const std::vector<T>& heavyPrefix,
                             const std::vector<T>& lightPrefix,
                             T average,
                             int n) {
    T localW = average * n;
    double dlocalW = (double)average * (double)n;
    int h_size = heavy.size();
    int l_size = heavy.size();

    int a = 1;
    int b = std::min(n, h_size);

    for (size_t i = 0; i < heavy.size(); ++i) {
      /* fmt::println("heavy[{}] = {}", i, heavy[i]); */
    }
    /* fmt::println("n = {}, localW = {}", n, localW); */

    uint32_t it = 0;
    uint32_t maxIt = static_cast<uint32_t>(std::ceil(std::log2(static_cast<float>(n))));
      
    while (true) {
        it += 1;
        /* fmt::println("Iteration: {}", it); */

        const int j = (a + b) / 2; // impl floor
        const int i = n - j;

        /* fmt::println("pivot = {}", j); */
        /* fmt::println("looking in [{}...{}]", a,b); */

        if (j >= heavyPrefix.size()
            || i < 1) {
          b = j - 1;
          /* fmt::println("out of range (to much heavy or to little light)"); */
          continue;
        }
        if (j < 1 || i >= lightPrefix.size()) {
          /* fmt::println("out of range (to little heavy or to much light)"); */
          a = j + 1;
          continue;
        }

        T l_prefix = lightPrefix.at(i - 1);
        T h_prefix = heavyPrefix.at(j - 1);
        T sigma = l_prefix + h_prefix;

        double dsigma = ((double)l_prefix ) + ((double)h_prefix);
        

        /* fmt::println("lightPrefix = {}, heavyPrefix = {} => sigma = {}", l_prefix, h_prefix, sigma); */
        if (dsigma <= dlocalW) {
            /* fmt::println("sigma <= local"); */
            /* fmt::println("next heavy = {} => {}", heavy.at(j), sigma + heavy.at(j)); */
            T next = heavy.at(j);
            double dnext = (double)heavy.at(j);

            if ((dsigma - dlocalW) + dnext > 0) {
                /* std::cout << sigma / localW << std::endl; */
                return {j, i, (float)(dnext + (dsigma - dlocalW)), sigma};
            } else {
                a = j + 1;
            }
        } else {
            /* fmt::println("sigma > local"); */
            b = j - 1;
        }
        if (it > maxIt * 2) {
          fmt::println("a = {}, b = {}, i = {}, j = {}, n = {}", a, b, i, j, n);
          fmt::println("(double) next = {}, sigma = {}, local = {}, s-l = {},\no = {}", heavy.at(j), dsigma, dlocalW, dsigma - dlocalW,
              ((double)heavy.at(j)) + (dsigma - dlocalW));
          fmt::println("(float)  next = {}, sigma = {}, local = {}, s-l = {},\no = {}", heavy.at(j), sigma, localW, sigma - localW,
              heavy.at(j) + (sigma - localW));
          /* std::cout << heavy.at(j) + (sigma - localW) << std::endl; */
        }

        /*if (it > maxIt * 2) {*/
        /*      T next = heavy.at(j);*/
        /*      return {j, i, next + (sigma - localW), sigma};*/
        /*}*/
    }

}

} // namespace wrs::cpu::stable
