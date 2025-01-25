#include "./test.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "src/renderdoc.hpp"
#include "src/wrs/algorithm/hs/HSTRepr.hpp"
#include "src/wrs/algorithm/pack/simd/SimdPack.hpp"
#include "src/wrs/gen/weight_generator.h"
#include "src/wrs/memory/FallbackResource.hpp"
#include "src/wrs/memory/SafeResource.hpp"
#include "src/wrs/memory/StackResource.hpp"
#include "src/wrs/reference/partition.hpp"
#include "src/wrs/reference/prefix_sum.hpp"
#include "src/wrs/reference/reduce.hpp"
#include "src/wrs/reference/split.hpp"
#include "src/wrs/test/is_alias_table.hpp"
#include "src/wrs/test/test.hpp"
#include "src/wrs/types/alias_table.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/base.h>
#include <fmt/format.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tuple>

#include "./HSTSampling.hpp"

using namespace wrs;
using namespace wrs::test;

using Algorithm = HSTSampling;
using Buffers = Algorithm::Buffers;

struct TestCase {
    glsl::uint workgroupSize;
    glsl::uint N;
    Distribution distribution;
    glsl::uint S;
    uint32_t iterations;
};

static constexpr TestCase TEST_CASES[] = {
    //
    TestCase{
        .workgroupSize = 512,
        .N = 1024 * 2048,
        .distribution = wrs::Distribution::SEEDED_RANDOM_UNIFORM,
        .S = 1024 * 2048,
        .iterations = 1,
    },
    /* TestCase{ */
    /*     .workgroupSize = 512, */
    /*     .N = 32, */
    /*     .distribution = wrs::Distribution::UNIFORM, */
    /*     .S = 10000, */
    /*     .iterations = 1, */
    /* }, */
};

static std::tuple<Buffers, Buffers> allocateBuffers(const TestContext& context) {
    glsl::uint maxN = 0;
    for (const auto& testCase : TEST_CASES) {
        maxN = std::max(maxN, testCase.N);
    }

    Buffers stage =
        Buffers::allocate(context.alloc, merian::MemoryMappingType::HOST_ACCESS_RANDOM, maxN);
    Buffers local = Buffers::allocate(context.alloc, merian::MemoryMappingType::NONE, maxN);

    return std::make_tuple(local, stage);
}

static void uploadTestCase(const vk::CommandBuffer cmd,
                           const Buffers& buffers,
                           const Buffers& stage,
                           std::span<const float> hst, 
                           std::size_t S) {
    Buffers::HstView stageView{stage.hst, hst.size()};
    Buffers::HstView localView{buffers.hst, hst.size()};
    stageView.upload(hst);
    stageView.copyTo(cmd, localView);
    localView.expectComputeRead(cmd);

    {
      glsl::uint* mapped = stage.samples->get_memory()->map_as<glsl::uint>();
      mapped[hst.size()] = S;
      stage.samples->get_memory()->unmap();
      Buffers::SamplesView stageView{stage.samples, hst.size() + 1};
      Buffers::SamplesView localView{buffers.samples, hst.size() + 1};
      stageView.expectHostWrite();
      stageView.copyTo(cmd, localView);
      localView.expectComputeRead(cmd);

    }
}

static void
downloadToStage(vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage, std::size_t N) {
    hst::HSTRepr repr{N};
    std::size_t entries = repr.size() + 1;
    Buffers::SamplesView stageView{stage.samples, entries};
    Buffers::SamplesView localView{buffers.samples, entries};
    localView.expectComputeWrite();
    localView.copyTo(cmd, stageView);
    stageView.expectHostRead(cmd);
}

struct Results {
    std::pmr::vector<glsl::uint> samples;
};
static Results
downloadFromStage(Buffers& stage, std::size_t N, std::pmr::memory_resource* resource) {
    hst::HSTRepr repr{N};
    std::size_t entries = repr.size() + 1;
    Buffers::SamplesView stageView{stage.samples, entries};
    auto samples = stageView.download<glsl::uint, wrs::pmr_alloc<glsl::uint>>(resource);
    return Results{
        .samples = std::move(samples),
    };
};

static bool runTestCase(const TestContext& context,
                        const TestCase& testCase,
                        Buffers& buffers,
                        Buffers& stage,
                        std::pmr::memory_resource* resource) {
    std::string testName = fmt::format("{{workgroupSize={},N={},S={}}}", testCase.workgroupSize,
                                       testCase.N, testCase.S);
    SPDLOG_INFO("Running test case:{}", testName);

    Algorithm kernel{context.context, testCase.workgroupSize};

    bool failed = false;
    for (size_t it = 0; it < testCase.iterations; ++it) {
        MERIAN_PROFILE_SCOPE(context.profiler, testName);
        context.queue->wait_idle();
        if (testCase.iterations > 1) {
            if (testCase.N > 1e6) {
                SPDLOG_INFO(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            } else {
                SPDLOG_DEBUG(
                    fmt::format("Testing iterations {} out of {}", it + 1, testCase.iterations));
            }
        }

        // 1. Generate input
        context.profiler->start("Generate test input");
        std::pmr::vector<float> hst =
            wrs::pmr::generate_weights<float>(testCase.distribution, testCase.N, resource);
        hst::HSTRepr repr{testCase.N};
        hst.resize(repr.size());
        for (const auto& level : repr.get()) {
            std::size_t invoc = level.numParents - (level.overlap ? 1u : 0u);
            for (std::size_t gid = 0; gid < invoc; ++gid) {
                std::size_t left = level.childOffset + gid * 2;
                std::size_t right = left + 1;
                float w = hst[left] + hst[right];
                hst[level.parentOffset + gid] = w;
            }
        }

        context.profiler->end();

        // 2. Begin recoding
        vk::CommandBuffer cmd = context.cmdPool->create_and_begin();
        std::string recordingLabel = fmt::format("Recording : {}", testName);
        context.profiler->start(recordingLabel);
        context.profiler->cmd_start(cmd, recordingLabel);

        // 3. Upload test case indices
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Upload test case");
            SPDLOG_DEBUG("Uploading test case...");
            uploadTestCase(cmd, buffers, stage, hst, testCase.S);
        }

        // 4. Run test case
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Execute algorithm");
            SPDLOG_DEBUG("Execute algorithm");
            kernel.run(cmd, buffers, testCase.N);
        }

        // 5. Download results to stage
        {
            MERIAN_PROFILE_SCOPE_GPU(context.profiler, cmd, "Download results to stage");
            SPDLOG_DEBUG("Downloading results to stage...");
            downloadToStage(cmd, buffers, stage, testCase.N);
        }

        // 6. Submit to device
        context.profiler->end();
        context.profiler->cmd_end(cmd);
        SPDLOG_DEBUG("Submitting to device...");
        cmd.end();
        context.queue->submit_wait(cmd);

        // 7. Download from stage
        context.profiler->start("Download results from stage");
        SPDLOG_DEBUG("Downloading results from stage...");
        Results results = downloadFromStage(stage, testCase.N, resource);
        context.profiler->end();

        // 7. Test results
        {
            MERIAN_PROFILE_SCOPE(context.profiler, "Testing results");
            SPDLOG_DEBUG("Testing results");
            if (testCase.N < 1024)  {
              for(std::size_t i = 0; i < results.samples.size(); ++i) {
                fmt::println("[{}]: {}", i, results.samples[i]);
              }
            }
        }
        context.profiler->collect(true, true);
    }
    return failed;
}

void wrs::test::hst_sampling::test(const merian::ContextHandle& context) {
    SPDLOG_INFO("Testing TODO algorithm");

    const TestContext testContext = setupTestContext(context);

    SPDLOG_DEBUG("Allocating buffers");
    auto [buffers, stage] = allocateBuffers(testContext);

    wrs::memory::StackResource stackResource{4096 * 2048};
    wrs::memory::FallbackResource fallbackResource{&stackResource};
    wrs::memory::SafeResource safeResource{&fallbackResource};

    std::pmr::memory_resource* resource = &safeResource;

    uint32_t failCount = 0;
    for (const auto& testCase : TEST_CASES) {
        renderdoc::startCapture();
        runTestCase(testContext, testCase, buffers, stage, resource);
        renderdoc::stopCapture();
        stackResource.reset();
    }

    testContext.profiler->collect(true, true);
    SPDLOG_INFO(fmt::format("Profiler results: \n{}",
                            merian::Profiler::get_report_str(testContext.profiler->get_report())));

    if (failCount == 0) {
        SPDLOG_INFO("All tests passed");
    } else {
        SPDLOG_ERROR(fmt::format("Failed {} out of {} tests", failCount,
                                 sizeof(TEST_CASES) / sizeof(TestCase)));
    }
}
