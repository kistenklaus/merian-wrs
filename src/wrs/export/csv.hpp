#include "src/wrs/why.hpp"
#include <array>
#include <fmt/format.h> // Include fmt library for formatting
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <array>
#include <fmt/core.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace wrs::exp {

template <std::size_t HeaderCount> class CSVWriter {
  public:
    CSVWriter(const std::array<std::string, HeaderCount>& headers,
              const std::string& filePath,
              char separator = ',')
        : file(filePath), separator(separator), buffer(), bufferLimit(1024 * 1024) // 1 MB buffer
    {
        buffer.reserve(bufferLimit); // Reserve space in the buffer to reduce allocations

        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + filePath);
        }

        if (separator == '\n') {
            throw std::invalid_argument("Invalid separator.");
        }

        writeHeaders(headers);
    }

    ~CSVWriter() {
        flushBuffer(); // Ensure the custom buffer is flushed before destruction
    }

    template <typename... Args>
        requires((std::is_floating_point_v<std::remove_reference_t<Args>> ||
                  std::is_convertible_v<std::remove_reference_t<Args>, std::string> ||
                  std::is_integral_v<std::remove_reference_t<Args>>) &&
                 ...)
    void pushRow(const Args&... args) {
        static_assert(sizeof...(Args) == HeaderCount,
                      "Row size must match the number of headers at compile time.");

        std::size_t index = 0;

        // Estimate the size of the new row and flush if necessary
        std::size_t estimatedRowSize = 0;
        ((estimatedRowSize += fmt::formatted_size("{}", args) + 1),
         ...); // +1 for separator or newline
        if (buffer.size() + estimatedRowSize >= bufferLimit) {
            flushBuffer();
        }

        (([&]() {
             buffer.append(fmt::format("{}", args));
             if (index++ < HeaderCount - 1) {
                 buffer.push_back(separator);
             }
         }()),
         ...);
        buffer.push_back('\n');
    }

    template <typename Tuple> void pushTupleRow(const Tuple& tuple) {
        static_assert(std::tuple_size<Tuple>::value == HeaderCount,
                      "Tuple size must match the number of headers.");
        std::apply([this](auto&&... args) { this->pushRow(std::forward<decltype(args)>(args)...); },
                   tuple);
    }

    template <typename T> void unsafePushValue(const T value, bool lastEntry) {
      std::size_t estimatedEntrySize = fmt::formatted_size("{}", value) + 1;
      if (buffer.size() + estimatedEntrySize >= bufferLimit) {
        flushBuffer();
      }
      buffer.append(fmt::format("{}", value));
      if (!lastEntry) {
        buffer.push_back(separator);
      }
    }

    void unsafeEndRow() {
      if (buffer.size() + 1 >= bufferLimit) {
        flushBuffer();
      }
      buffer.push_back('\n');
    }

  private:
    std::ofstream file;
    char separator;
    std::string buffer;
    const std::size_t bufferLimit;

    void writeHeaders(const std::array<std::string, HeaderCount>& headers) {
        for (std::size_t i = 0; i < headers.size(); ++i) {
            if (headers[i].empty() || headers[i].find_first_of('\n') != std::string::npos) {
                throw std::invalid_argument("Invalid header: " + headers[i]);
            }
            buffer.append(headers[i]);
            if (i < headers.size() - 1) {
                buffer.push_back(separator);
            }
        }
        buffer.push_back('\n');
    }

    void flushBuffer() {
        if (!buffer.empty()) {
            file.write(buffer.data(), buffer.size());
            if (file.fail()) {
                throw std::ios_base::failure("Failed to write to the file.");
            }
            buffer.clear();
        }
    }
};

} // namespace wrs::exp

/* static_assert(std::is_same_v<std::remove_reference_t<std::string>, std::string>); */
