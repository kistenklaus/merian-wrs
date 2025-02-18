#pragma once

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "src/wrs/algorithm/histogram/atomic/AtomicHistogram.hpp"
#include "src/wrs/algorithm/rmse/mse/MeanSquaredError.hpp"
#include "src/wrs/eval/histogram.hpp"
#include "src/wrs/reference/inverse_alias_table.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/types/glsl.hpp"
#include <cmath>
#include <concepts>
#include <fmt/base.h>
#include <glm/ext/scalar_constants.hpp>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <spdlog/spdlog.h>
#include <tuple>
#include <type_traits>
namespace wrs::eval {

template <std::floating_point E,
          std::integral I,
          wrs::typed_allocator<I> Allocator = std::allocator<I>>
E rmse(std::span<const E> weights,
       std::span<const I> samples,
       std::optional<E> totalWeight = std::nullopt,
       const Allocator& alloc = {}) {
    assert(!weights.empty());
    assert(!samples.empty());
    if (!totalWeight.has_value()) {
        totalWeight = wrs::reference::kahan_reduction<E>(weights);
    }
    assert(totalWeight.has_value());

    if (*totalWeight < std::numeric_limits<E>::epsilon()) {
        SPDLOG_WARN(fmt::format(
            "Computing rmse_curve with a very small total weight might lead to invalid results.\n",
            "totalWeight = {}", *totalWeight));
    }

    std::vector<I, Allocator> histogram = wrs::eval::histogram(samples, weights.size(), alloc);

    std::size_t n = weights.size();
    E c = E{};
    E rmse{};
    for (size_t i = 0; i < n; ++i) {
        E w = weights[i];
        assert(w >= 0);
        const E expected = weights[i] / *totalWeight;
        const E observed = static_cast<E>(histogram[i]) / samples.size();
        const E diff = expected - observed;
        /* const E diff = (w * n - histogram[i] * totalWeight.value()) / (*totalWeight * n); */
        const E diff2 = diff * diff;
        const E term = diff2 - c;
        const E temp = rmse + term;
        c = (temp - rmse) - term;
        rmse = temp;
    }
    rmse = sqrt(rmse / static_cast<E>(weights.size()));
    return rmse;
}

template <std::floating_point E,
          std::integral I,
          wrs::typed_allocator<I> Allocator = std::allocator<I>>
E histogram_rmse(std::span<const E> weights,
                 std::span<const I> histogram,
                 std::optional<E> totalWeight = std::nullopt,
                 [[maybe_unused]] const Allocator& alloc = {}) {
    assert(!weights.empty());
    assert(!histogram.empty());
    if (!totalWeight.has_value()) {
        totalWeight = wrs::reference::kahan_reduction<E>(weights);
    }
    assert(totalWeight.has_value());

    if (*totalWeight < std::numeric_limits<E>::epsilon()) {
        SPDLOG_WARN(fmt::format(
            "Computing rmse_curve with a very small total weight might lead to invalid results.\n",
            "totalWeight = {}", *totalWeight));
    }

    std::size_t n = weights.size();
    E c = E{};
    E rmse{};
    for (size_t i = 0; i < n; ++i) {
        E w = weights[i];
        assert(w >= 0);
        const E expected = weights[i] / *totalWeight;
        const E observed = static_cast<E>(histogram[i]) / n;
        const E diff = expected - observed;
        /* const E diff = (w * n - histogram[i] * totalWeight.value()) / (*totalWeight * n); */
        const E diff2 = diff * diff;
        const E term = diff2 - c;
        const E temp = rmse + term;
        c = (temp - rmse) - term;
        rmse = temp;
    }
    rmse = sqrt(rmse / static_cast<E>(weights.size()));
    return rmse;
}

template <std::floating_point E,
          std::integral I,
          std::ranges::input_range Scale,
          wrs::generic_allocator Allocator = std::allocator<E>>
    requires std::is_integral_v<std::ranges::range_value_t<Scale>>
std::vector<std::tuple<std::ranges::range_value_t<Scale>, E>,
            typename std::allocator_traits<Allocator>::template rebind_alloc<
                std::tuple<std::ranges::range_value_t<Scale>, E>>>
rmse_curve(std::span<const E> weights,
           std::span<const I> samples,
           const Scale xScale,
           std::optional<E> totalWeight = std::nullopt,
           const Allocator& alloc = Allocator{}) {

    using Size = std::ranges::range_value_t<Scale>;

    assert(!weights.empty());
    assert(!samples.empty());
    assert(!std::ranges::empty(xScale));

    using HistogramAllocator =
        typename std::allocator_traits<Allocator>::template rebind_alloc<Size>;
    using ResultAllocator =
        typename std::allocator_traits<Allocator>::template rebind_alloc<std::tuple<Size, E>>;

    if (!totalWeight.has_value()) {
        totalWeight = wrs::reference::kahan_reduction<E>(weights);
    }

    if (*totalWeight < std::numeric_limits<E>::epsilon()) {
        SPDLOG_WARN(fmt::format(
            "Computing rmse_curve with a very small total weight might lead to invalid results.\n",
            "totalWeight = {}", *totalWeight));
    }

    Size scaleSize = 0;
    if constexpr (std::ranges::sized_range<Scale>) {
        scaleSize = std::ranges::size(xScale);
    } else {
        for (const auto& _ : xScale) {
            ++scaleSize;
        }
    }

    std::vector<Size, HistogramAllocator> histogram(weights.size(), 0, alloc);

    std::vector<std::tuple<Size, E>, ResultAllocator> results(alloc);
    results.reserve(scaleSize);

    Size lastN = 0;

    for (const auto& n : xScale) {
        assert(n > 0 && "Sample size (n) must be greater than 0.");
        assert(n <= samples.size());
        assert(n >= lastN && "xScale must be monotone");

        for (Size i = lastN; i < n; ++i) {
            assert(samples[i] < weights.size());
            histogram[samples[i]]++;
        }

        assert(std::accumulate(histogram.begin(), histogram.end(), 0) == n);

        lastN = n;

        E rmse = E{};
        E c = E{};

        E expectedAverage = static_cast<E>(n) / *totalWeight;

        for (Size i = 0; i < weights.size(); ++i) {
            E w = weights[i];
            assert(w >= 0);
            const E expected = weights[i] / *totalWeight;
            const E observed = static_cast<E>(histogram[i]) / n;
            const E diff = expected - observed;
            /* const E diff = (w * n - histogram[i] * totalWeight.value()) / (*totalWeight * n); */
            const E diff2 = diff * diff;
            const E term = diff2 - c;
            const E temp = rmse + term;
            c = (temp - rmse) - term;
            rmse = temp;
        }
        rmse = sqrt(rmse / static_cast<E>(weights.size()));

        results.emplace_back(n, rmse);
    }

    return results;
}

// Building pattern for a RMSE curve, which
// allows working with large sample counts, which
// are provided in multiple calls. This reduces
// the total amount of memory required to the largest submission.
//
// For example, if we want to compute the RMSE over 1e12 elements,
// that would be 1000Gb of data just for the samples range, which is obviously
// not an option and the OS will kill the application immediately assuming
// this to be a bad allocation (error might occure during write not allocate, because of lazy page
// allocation).
//
// E: type used in the RMSE computation (E should generally have higher or equal precision than W)
// W: type of the weights
// I: type of the samples section ranges
// Scale: the x scale of the RMSE curve to compute!
template <std::floating_point E, std::floating_point W, std::integral I>
struct RMSECurveSectionedBuilder {

    template <std::ranges::input_range Scale>
        requires std::is_integral_v<std::ranges::range_value_t<Scale>>
    explicit RMSECurveSectionedBuilder(const std::span<const W> weights, const Scale& scale)
        : m_probabilities(wrs::reference::weights_to_probabilities<W, E>(weights)),
          m_histogram(weights.size(), 0), m_s(0), m_k(0) {

        auto scaleIt = std::ranges::begin(scale);
        const auto scaleEnd = std::ranges::end(scale);
        while (scaleIt != scaleEnd) {
            m_rmseCurve.push_back(std::make_tuple(static_cast<uint64_t>(*scaleIt), 0));
            scaleIt++;
        }
    }

    void consume(std::span<const I> samplesSection) {
        std::span<const I> todo = samplesSection;

        while (!todo.empty()) {
            if (m_k >= m_rmseCurve.size()) {
                SPDLOG_ERROR("Invalid index k");
                break;
            }
            uint64_t next = std::get<0>(m_rmseCurve[m_k]);
            uint64_t curr = m_s;
            uint64_t required = next - curr;
            if (todo.size() <= required) {
                appendHistogram(todo);
                todo = {}; // empty
            } else {
                appendHistogram(todo.subspan(0, required));
                std::span<const I> remaining = todo.subspan(required);
                todo = remaining;
            }
            if (m_s == next) {
                computeRmse();
            }
        }
    }

    void appendHistogram(std::span<const I> samples) {
        for (const I i : samples) {
            if (i >= m_histogram.size()) {
                SPDLOG_ERROR("Invalid sample : {}", i);
            }
            m_histogram[i]++;
        }
        m_s += samples.size();
        if (m_k >= m_rmseCurve.size()) {
            SPDLOG_ERROR("Invalid k index");
        } else {
            assert(m_s <= std::get<0>(m_rmseCurve[m_k]));
        }
    }

    void computeRmse() {
        std::size_t n = m_probabilities.size();
        E rmse = E{};
        E c = E{};

        if (m_k >= m_rmseCurve.size()) {
            SPDLOG_ERROR("Invalid k index");
        }
        uint64_t s = std::get<0>(m_rmseCurve[m_k]);
        /*uint64_t histogramCount = std::accumulate(m_histogram.begin(), m_histogram.end(), 0ull);*/
        /*assert(histogramCount == s);*/
        assert(s == m_s);

        for (std::size_t i = 0; i < n; ++i) {

            const E expected = m_probabilities[i];
            const E observed = static_cast<E>(m_histogram[i]) / m_s;
            const E diff = expected - observed;
            /* const E diff = (w * n - histogram[i] * totalWeight.value()) / (*totalWeight * n); */
            const E diff2 = diff * diff;
            const E term = diff2 - c;
            const E temp = rmse + term;
            c = (temp - rmse) - term;
            rmse = temp;
        }
        rmse = std::sqrt(rmse / static_cast<E>(n));
        SPDLOG_DEBUG("Partial RMSE Curve result done: S={} -> {}", m_s, rmse);
        std::get<1>(m_rmseCurve[m_k++]) = rmse;
    }

    std::span<const std::tuple<uint64_t, E>> get() {
        return m_rmseCurve;
    }

    const std::vector<E> m_probabilities; // [i] = w / mean;   N elements

    std::vector<uint64_t> m_histogram; // N elements
    uint64_t m_s;                      // amount of sample in the histogram

    std::vector<std::tuple<uint64_t, E>> m_rmseCurve; // |Scale| elements.
    std::size_t m_k; // index of rmse entry that is currently beeing build!
};

// GPU accelerated RMSE curve builder pattern
struct RMSECurveAcceleratedBuilder {

    template <std::ranges::input_range Scale>
        requires std::is_integral_v<std::ranges::range_value_t<Scale>>
    explicit RMSECurveAcceleratedBuilder(const merian::ContextHandle& context,
                                         const merian::ShaderCompilerHandle& shaderCompiler,
                                         merian::BufferHandle weights,
                                         float totalWeight,
                                         glsl::uint N,
                                         const Scale& scale)
        : m_histogramKernel(context, shaderCompiler), m_rmeKernel(context, shaderCompiler), m_s(0),
          m_weights(weights), m_n(N), m_totalWeight(totalWeight), m_k(0) {

        auto scaleIt = std::ranges::begin(scale);
        const auto scaleEnd = std::ranges::end(scale);
        while (scaleIt != scaleEnd) {
            m_rmseCurve.push_back(std::make_tuple(static_cast<uint64_t>(*scaleIt), 0));
            scaleIt++;
        }

        auto resources = context->get_extension<merian::ExtensionResources>();
        merian::ResourceAllocatorHandle alloc = resources->resource_allocator();
        m_histogram = alloc->createBuffer(wrs::AtomicHistogramBuffers::HistogramLayout::size(N),
                                          vk::BufferUsageFlagBits::eStorageBuffer |
                                              vk::BufferUsageFlagBits::eTransferSrc,
                                          merian::MemoryMappingType::NONE);

        m_histogramStage = alloc->createBuffer(
            wrs::AtomicHistogramBuffers::HistogramLayout::size(N),
            vk::BufferUsageFlagBits::eTransferDst, merian::MemoryMappingType::HOST_ACCESS_RANDOM);

        m_mse = alloc->createBuffer(
            wrs::MeanSquaredError::Buffers::MseLayout::size(m_rmseCurve.size()),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            merian::MemoryMappingType::NONE);

        m_mseStage = alloc->createBuffer(
            wrs::MeanSquaredError::Buffers::MseLayout::size(m_rmseCurve.size()),
            vk::BufferUsageFlagBits::eTransferDst, merian::MemoryMappingType::HOST_ACCESS_RANDOM);
    }

    void
    consume(const merian::CommandBufferHandle& cmd, merian::BufferHandle samples, glsl::uint s) {
        if (m_k == m_rmseCurve.size()) {
            return;
        }
        glsl::uint offset = 0;
        int64_t count = s;

        while (count > 0) {
            if (m_k >= m_rmseCurve.size()) {
                break;
            }
            uint64_t next = std::get<0>(m_rmseCurve[m_k]);
            uint64_t curr = m_s;
            int64_t required = next - curr;
            assert(required >= 0);
            if (count <= required) {
                appendHistogram(cmd, samples, offset, count);
                count = 0;
            } else {
                appendHistogram(cmd, samples, offset, required);
                offset += required;
                count -= required;
            }
            if (m_s == next) {
                computeMse(cmd);
            }
        }
        if (m_k == m_rmseCurve.size()) {
            downloadMSE_toStage(cmd);
        }
    }

    void appendHistogram(const merian::CommandBufferHandle cmd,
                         merian::BufferHandle samples,
                         glsl::uint offset,
                         glsl::uint count) {
        SPDLOG_DEBUG("Appending to histogram");
        m_s += count;
        wrs::AtomicHistogramBuffers buffers;
        buffers.samples = samples;
        buffers.histogram = m_histogram;
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     samples->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                             vk::AccessFlagBits::eShaderRead));
        m_histogramKernel.run(cmd, buffers, offset, count);

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     m_histogram->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                 vk::AccessFlagBits::eShaderWrite));
    }

    void computeMse(const merian::CommandBufferHandle cmd) {
        SPDLOG_DEBUG("Computing MSE");
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     m_histogram->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                 vk::AccessFlagBits::eShaderRead));

        wrs::MeanSquaredError::Buffers buffers;
        buffers.histogram = m_histogram;
        buffers.weights = m_weights;
        buffers.mse = m_mse;
        m_rmeKernel.run(cmd, buffers, m_k, static_cast<float>(m_s), m_n, m_totalWeight);

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     m_histogram->buffer_barrier(vk::AccessFlagBits::eShaderRead,
                                                 vk::AccessFlagBits::eShaderWrite));
        m_k++;
    }

    void downloadMSE_toStage(const merian::CommandBufferHandle& cmd) {
        SPDLOG_DEBUG("Downloading to Stage");
        wrs::MeanSquaredError::Buffers::MseView stageView{m_mseStage, m_rmseCurve.size()};
        wrs::MeanSquaredError::Buffers::MseView localView{m_mse, m_rmseCurve.size()};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);

        {
            wrs::AtomicHistogram::Buffers::HistogramView stageView{m_histogramStage, m_n};
            wrs::AtomicHistogram::Buffers::HistogramView localView{m_histogram, m_n};
            localView.expectComputeWrite();
            localView.copyTo(cmd, stageView);
            stageView.expectHostRead(cmd);
        }
    }

    std::span<const std::tuple<uint64_t, float>> get() {
        wrs::MeanSquaredError::Buffers::MseView stageView{m_mseStage, m_rmseCurve.size()};
        std::vector<float> mseCurve = stageView.download<float>();
        for (std::size_t i = 0; i < mseCurve.size(); ++i) {
            float mse = mseCurve[i];
            float rmse = std::sqrt(mse / static_cast<float>(m_n));
            std::get<1>(m_rmseCurve[i]) = rmse;
        }
        {
            /* wrs::AtomicHistogram::Buffers::HistogramView stageView{m_histogramStage, m_n}; */
            /* const auto histogram = stageView.download<glsl::uint64>(); */
            /* for (std::size_t i = 0; i < histogram.size(); ++i) { */
            /*     fmt::println("[{}]: {}", i, histogram[i]); */
            /* } */
        }

        return m_rmseCurve;
    }

    wrs::AtomicHistogram m_histogramKernel;
    wrs::MeanSquaredError m_rmeKernel;

    merian::BufferHandle m_histogram;
    merian::BufferHandle m_histogramStage;
    uint64_t m_s; // amount of sample in the histogram

    merian::BufferHandle m_mse;
    merian::BufferHandle m_mseStage;

    merian::BufferHandle m_weights;
    glsl::uint m_n;
    float m_totalWeight;

    std::vector<std::tuple<uint64_t, float>> m_rmseCurve; // |Scale| elements.
    std::size_t m_k; // index of rmse entry that is currently beeing build!
};

namespace pmr {

template <std::floating_point E, std::integral I, std::ranges::input_range Scale>
    requires std::is_integral_v<std::ranges::range_value_t<Scale>>
std::pmr::vector<std::tuple<std::ranges::range_value_t<Scale>, E>>
rmse_curve(std::span<const E> weights,
           std::span<const I> samples,
           const Scale xScale,
           std::optional<E> totalWeight = std::nullopt,
           const std::pmr::polymorphic_allocator<void>& alloc = {}) {
    return wrs::eval::rmse_curve<E, I, Scale, std::pmr::polymorphic_allocator<void>>(
        weights, samples, xScale, totalWeight, alloc);
}
} // namespace pmr

} // namespace wrs::eval
