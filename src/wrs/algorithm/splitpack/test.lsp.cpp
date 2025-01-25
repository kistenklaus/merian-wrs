#include "./test.hpp"
#include "./SplitPack.hpp"
#include <algorithm>
#include <cstring>

using namespace wrs;

using Algorithm = SplitPack;
using Buffers = Algorithm::Buffers;

namespace wrs::test::splitpack {

void uploadTestCase(const vk::CommandBuffer cmd,
                    const Buffers& buffers,
                    const Buffers& stage,
                    std::span<const float> weights,
                    std::span<const glsl::uint> lightIndices,
                    std::span<const glsl::uint> heavyIndices,
                    std::span<const float> lightPrefix,
                    std::span<const float> heavyPrefix,
                    const float mean,
                    std::pmr::memory_resource* resource) {
    {
        Buffers::WeightsView stageView{stage.weights, weights.size()};
        Buffers::WeightsView localView{buffers.weights, weights.size()};
        stageView.upload(weights);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    glsl::uint heavyCount = heavyIndices.size();
    assert(heavyCount == heavyPrefix.size());
    glsl::uint lightCount = weights.size() - heavyCount;
    assert(lightCount == lightPrefix.size());
    assert(lightCount == lightIndices.size());
    {
        Buffers::PartitionIndicesView stageView{stage.partitionIndices, weights.size()};
        Buffers::PartitionIndicesView localView{buffers.partitionIndices, weights.size()};
        std::pmr::vector<glsl::uint> storage{weights.size(), resource};
        std::memcpy(storage.data(), heavyIndices.data(), heavyCount * sizeof(glsl::uint));
        std::memcpy(storage.data() + heavyCount, lightIndices.data(),
                    lightCount * sizeof(glsl::uint));
        std::reverse(storage.begin() + heavyCount, storage.end());
        stageView.attribute<"heavyCount">().template upload<glsl::uint>(heavyCount);
        stageView.attribute<"heavyLight">().template upload<glsl::uint>(storage);
        stageView.expectHostWrite();
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        Buffers::PartitionPrefixView stageView{stage.partitionPrefix, weights.size()};
        Buffers::PartitionPrefixView localView{buffers.partitionPrefix, weights.size()};

        std::pmr::vector<float> storage{weights.size(), resource};
        std::memcpy(storage.data(), heavyPrefix.data(), heavyCount * sizeof(float));
        std::memcpy(storage.data() + heavyCount, lightPrefix.data(), lightCount * sizeof(float));
        std::reverse(storage.begin() + heavyCount, storage.end());
        stageView.attribute<"heavyCount">().template upload<glsl::uint>(heavyCount);
        stageView.attribute<"heavyLight">().template upload<float>(storage);
        stageView.expectHostWrite();
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
    {
        Buffers::MeanView stageView{stage.mean};
        Buffers::MeanView localView{buffers.mean};
        stageView.upload<float>(mean);
        stageView.copyTo(cmd, localView);
        localView.expectComputeRead(cmd);
    }
}

void downloadToStage(
    vk::CommandBuffer cmd, Buffers& buffers, Buffers& stage, glsl::uint N, glsl::uint K) {

    {
        Buffers::AliasTableView stageView{stage.aliasTable, N};
        Buffers::AliasTableView localView{buffers.aliasTable, N};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
    {
        Buffers::SplitsView stageView{stage.splits, K};
        Buffers::SplitsView localView{buffers.splits, K};
        localView.expectComputeWrite();
        localView.copyTo(cmd, stageView);
        stageView.expectHostRead(cmd);
    }
}

Results
downloadFromStage(Buffers& stage, glsl::uint N, glsl::uint K, std::pmr::memory_resource* resource) {
    Results results = {
        .aliasTable = wrs::pmr::AliasTable<float, glsl::uint>{resource},
        .splits = std::pmr::vector<wrs::Split<float, glsl::uint>>{resource},
    };
    {
      Buffers::AliasTableView stageView{stage.aliasTable, N};
      using Entry = wrs::AliasTableEntry<float, wrs::glsl::uint>;
      results.aliasTable = stageView.download<Entry, wrs::pmr_alloc<Entry>>(resource);
    }
    {
      Buffers::SplitsView stageView{stage.splits, K};
      using Split = wrs::Split<float, wrs::glsl::uint>;
      results.splits = stageView.download<Split, wrs::pmr_alloc<Split>>(resource);
    }

    return results;
}

} // namespace wrs::test::splitpack
