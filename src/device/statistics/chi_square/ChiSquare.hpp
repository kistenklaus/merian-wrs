#pragma once

#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/statistics/chi_square/ChiSquareAllocFlags.hpp"
#include "src/device/statistics/chi_square/reduce/ChiSquareReduce.hpp"
#include "src/device/statistics/chi_square/reduce/ChiSquareReduceAllocFlags.hpp"
#include "src/device/statistics/histogram/Histogram.hpp"
#include "src/device/statistics/histogram/HistogramAllocFlags.hpp"
#include "vulkan/vulkan_enums.hpp"

namespace device {

struct ChiSquareBuffers {
    using Self = ChiSquareBuffers;
    static constexpr auto storageQualifier = host::glsl::StorageQualifier::std430;

    merian::BufferHandle samples;
    using SamplesLayout = host::layout::ArrayLayout<host::glsl::uint, storageQualifier>;
    using SamplesView = host::layout::BufferView<SamplesLayout>;

    merian::BufferHandle weights;
    using WeightsLayout = host::layout::ArrayLayout<float, storageQualifier>;
    using WeightsView = host::layout::BufferView<WeightsLayout>;

    merian::BufferHandle chiSquare;
    using ChiSquareLayout = host::layout::PrimitiveLayout<float, storageQualifier>;
    using ChiSquareView = host::layout::BufferView<ChiSquareLayout>;

    merian::BufferHandle m_histogram;

    static Self allocate(const merian::ResourceAllocatorHandle& alloc,
                         merian::MemoryMappingType memoryMapping,
                         host::glsl::uint N,
                         host::glsl::uint S,
                         ChiSquareAllocFlags allocFlags = ChiSquareAllocFlags::ALLOC_ALL) {

        Self buffers;
        ChiSquareReduceAllocFlags reduceFlags = ChiSquareReduceAllocFlags::ALLOC_NONE;
        if ((allocFlags & ChiSquareAllocFlags::ALLOC_CHI_SQUARE) != 0) {
            reduceFlags |= ChiSquareReduceAllocFlags::ALLOC_CHI_SQUARE;
        }
        if ((allocFlags & ChiSquareAllocFlags::ALLOC_WEIGHTS) != 0) {
            reduceFlags |= ChiSquareReduceAllocFlags::ALLOC_WEIGHTS;
        }

        ChiSquareReduce::Buffers reduceBuffers =
            ChiSquareReduce::Buffers::allocate(alloc, memoryMapping, N, reduceFlags);
        buffers.chiSquare = reduceBuffers.chiSquare;
        buffers.weights = reduceBuffers.weights;

        HistogramAllocFlags histogramAllocFlags = HistogramAllocFlags::ALLOC_HISTOGRAM;
        if ((allocFlags & ChiSquareAllocFlags::ALLOC_SAMPLES) != 0) {
            histogramAllocFlags |= HistogramAllocFlags::ALLOC_SAMPLES;
        }
        Histogram::Buffers histogramBuffers =
            Histogram::Buffers::allocate(alloc, memoryMapping, N, S, histogramAllocFlags);
        buffers.samples = histogramBuffers.samples;
        buffers.m_histogram = histogramBuffers.histogram;

        return buffers;
    }
};

struct ChiSquareConfig {
    const ChiSquareReduce::Config m_reduceConfig;
    const Histogram::Config m_histogramConfig;

    constexpr ChiSquareConfig() : m_reduceConfig(), m_histogramConfig() {}
};

class ChiSquare {
  public:
    using Buffers = ChiSquareBuffers;
    using Config = ChiSquareConfig;

    explicit ChiSquare(const merian::ContextHandle& context,
                       const merian::ShaderCompilerHandle& shaderCompiler,
                       const Config& config = {})
        : m_histogram(context, shaderCompiler, config.m_histogramConfig),
          m_reduce(context, shaderCompiler, config.m_reduceConfig) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint N,
             host::glsl::uint S,
             host::glsl::f32 totalWeight) const {
        Histogram::Buffers histogramBuffers;
        histogramBuffers.samples = buffers.samples;
        histogramBuffers.histogram = buffers.m_histogram;
        m_histogram.run(cmd, histogramBuffers, 0, S);

        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     buffers.m_histogram->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                                         vk::AccessFlagBits::eShaderRead));

        ChiSquareReduce::Buffers reduceBuffers;
        reduceBuffers.weights = buffers.weights;
        reduceBuffers.histogram = buffers.m_histogram;
        reduceBuffers.chiSquare = buffers.chiSquare;
        m_reduce.run(cmd, reduceBuffers, N, S, totalWeight);
    }

  private:
    Histogram m_histogram;
    ChiSquareReduce m_reduce;
};

} // namespace device
