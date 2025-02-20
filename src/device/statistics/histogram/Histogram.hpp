
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "src/device/statistics/histogram/atomic/AtomicHistogram.hpp"

namespace device {

using HistogramBuffers = AtomicHistogram::Buffers;
using HistogramConfig = AtomicHistogram::Config;

class Histogram {
  public:
    using Buffers = HistogramBuffers;
    using Config = HistogramConfig;
    using Method = AtomicHistogram;

    explicit Histogram(const merian::ContextHandle& context,
                       const merian::ShaderCompilerHandle& shaderCompiler,
                       Config config = {})
        : m_method(context, shaderCompiler, config) {}

    void run(const merian::CommandBufferHandle& cmd,
             const Buffers& buffers,
             host::glsl::uint offset,
             host::glsl::uint count) const {
        m_method.run(cmd, buffers, offset, count);
    }

  private:
    Method m_method;
};

} // namespace device
