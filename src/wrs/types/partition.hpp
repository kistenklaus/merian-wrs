#pragma once

#include "src/wrs/why.hpp"
#include <concepts>
#include <ranges>
#include <span>
#include <vector>
namespace wrs {

template <wrs::arithmetic T, std::ranges::contiguous_range ContigousRange>
    requires(std::same_as<std::ranges::range_value_t<ContigousRange>, T>)
class Partition {
  public:
    using difference_type = std::ranges::range_difference_t<ContigousRange>;
    Partition() = default;
    ~Partition() = default;
    Partition(ContigousRange storageHeavyLight, difference_type heavyCount)
        : m_storage(std::move(storageHeavyLight)), m_heavyCount(heavyCount) {}

    std::span<T> light() {
        return std::span(m_storage.begin() + m_heavyCount, m_storage.end());
    };

    std::span<T> heavy() {
        return std::span(m_storage.begin(), m_storage.begin() + m_heavyCount);
    };

    std::span<const T> light() const{
        return std::span(m_storage.begin() + m_heavyCount, m_storage.end());
    };

    std::span<const T> heavy() const {
        return std::span(m_storage.begin(), m_storage.begin() + m_heavyCount);
    };

    auto data() const {
      return m_storage.data();
    }

    std::span<const T> storage() const {
      return m_storage;
    };

    auto size_bytes() const {
      return std::ranges::size(m_storage) * sizeof(T);
    }

  private:
    difference_type m_heavyCount;
    ContigousRange m_storage;
};
static_assert(std::semiregular<Partition<float, std::vector<float>>>);

} // namespace wrs
