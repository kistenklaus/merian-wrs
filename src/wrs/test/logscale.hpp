#pragma once

#include "src/wrs/why.hpp"
#include <cassert>
#include <cmath>
#include <iterator>
#include <type_traits>
namespace wrs {

namespace internal {

template <std::floating_point T> class FloatLogScaleIterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    FloatLogScaleIterator(T logStart, T step, size_t index)
        : m_logStart(logStart), m_step(step), m_index(index) {}

    T operator*() const {
        return std::pow(10, m_logStart + m_index * m_step);
    }

    FloatLogScaleIterator& operator++() {
        ++m_index;
        return *this;
    }

    FloatLogScaleIterator operator++(int) {
        FloatLogScaleIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const FloatLogScaleIterator& other) const {
        return m_index == other.m_index;
    }

    bool operator!=(const FloatLogScaleIterator& other) const {
        return !(*this == other);
    }

  private:
    T m_logStart;
    T m_step;
    size_t m_index;
};

template <std::floating_point T> class FloatLogScaleRange {
  public:
    FloatLogScaleRange(T start, T end, size_t numPoints)
        : m_logStart(std::log10(start)),
          m_step((std::log10(end) - m_logStart) / static_cast<T>(numPoints - 1)),
          m_numPoints(numPoints) {
        assert(start > 0 && end > start && numPoints > 1);
    }

    FloatLogScaleIterator<T> begin() const {
        return FloatLogScaleIterator<T>(m_logStart, m_step, 0);
    }

    FloatLogScaleIterator<T> end() const {
        return FloatLogScaleIterator<T>(m_logStart, m_step, m_numPoints);
    }

  private:
    T m_logStart;
    T m_step;
    size_t m_numPoints;
};

// Integer-based logscale iterator
template <std::integral T> class IntLogScaleIterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    IntLogScaleIterator(T start, double multiplier, size_t index)
        : m_start(start), m_multiplier(multiplier), m_index(index), m_current(start) {
        if (index > 0) {
            m_current = static_cast<T>(start * std::pow(multiplier, index));
        }
    }

    T operator*() const {
        return m_current;
    }

    IntLogScaleIterator& operator++() {
        ++m_index;
        m_current = static_cast<T>(m_start * std::pow(m_multiplier, m_index));
        return *this;
    }

    IntLogScaleIterator operator++(int) {
        IntLogScaleIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const IntLogScaleIterator& other) const {
        return m_index == other.m_index;
    }

    bool operator!=(const IntLogScaleIterator& other) const {
        return !(*this == other);
    }

  private:
    T m_start;
    double m_multiplier;
    size_t m_index;
    T m_current;
};

template <std::integral T> class IntLogScaleRange {
  public:
    IntLogScaleRange(T start, T end, size_t numPoints)
        : m_start(start),
          m_multiplier(std::pow(static_cast<double>(end) / start, 1.0 / (numPoints - 1))),
          m_numPoints(numPoints) {
        assert(start > 0 && end > start && numPoints > 1);
    }

    IntLogScaleIterator<T> begin() const {
        return IntLogScaleIterator<T>(m_start, m_multiplier, 0);
    }

    IntLogScaleIterator<T> end() const {
        return IntLogScaleIterator<T>(m_start, m_multiplier, m_numPoints);
    }

  private:
    T m_start;
    double m_multiplier;
    size_t m_numPoints;
};

} // namespace internal

template <wrs::arithmetic T> auto log10scale(T start, T end, size_t numPoints) {
    if constexpr (std::is_floating_point_v<T>) {
        return internal::FloatLogScaleRange<T>(start, end, numPoints);
    } else if constexpr (std::is_integral_v<T>) {
        return internal::IntLogScaleRange<T>(start, end, numPoints);
    } else {
        static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                      "Unsupported type for log10scale");
    }
}

} // namespace wrs
