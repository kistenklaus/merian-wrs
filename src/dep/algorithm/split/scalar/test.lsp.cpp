#include "./ScalarSplit.hpp"
#include <vector>

using weight_t = float;
using Buffers = wrs::ScalarSplit::Buffers;

namespace wrs::test::scalar_split {

std::pmr::vector<Buffers::Split<weight_t>> downloadResultsFromStage(Buffers& stage, uint32_t K, std::pmr::memory_resource* resource) {
     Buffers::SplitsView stageView{stage.splits, K}; 
     return stageView.template download<Buffers::Split<weight_t>, wrs::pmr_alloc<Buffers::Split<weight_t>>>(resource); 
}

}
