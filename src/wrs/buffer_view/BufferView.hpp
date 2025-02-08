

#include "src/wrs/types/glsl.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <fmt/base.h>
#include <string_view>
#include <tuple>
#include <type_traits>
namespace wrs {

template <std::size_t N> struct StaticString {
    char value[N];

    consteval StaticString(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }

    consteval operator std::string_view() const {
        return std::string_view(value, N - 1); // Exclude null terminator
    }
};

template <typename First, typename... Rest>
constexpr bool has_pointer_in_middle = false; // Default: No issue

template <typename First, typename Second, typename... Rest>
constexpr bool has_pointer_in_middle<First, Second, Rest...> =
    (std::is_pointer_v<First> && !std::is_pointer_v<Second>) || // Pointer followed by non-pointer
    has_pointer_in_middle<Second, Rest...>;                     // Check remaining elements

template <StaticString Name, typename... Layout> struct BufferView {

    static constexpr bool contains_pointer = (std::is_pointer_v<Layout> || ...);

    template <typename First, typename... Rest>
    static constexpr bool contains_pointer_except_last =
        (std::is_pointer_v<First> || ... || std::is_pointer_v<Rest>);

    template <typename Last>
    static constexpr bool contains_pointer_except_last<Last> =
        false; // Base case (only last type remains)

    static_assert(contains_pointer_except_last<Layout...>);

    // Base case: For all other types, they are considered sized unless they're pointers
    template <typename T> static constexpr bool is_sized = !std::is_pointer_v<T>;

    // Special case: If T is a BufferView<U...>, we check if its size is also known
    template <StaticString X, typename... Ts>
    static constexpr bool is_sized<BufferView<X, Ts...>> = BufferView<Name, Ts...>::is_sized_buffer;

    // Recursively check if a BufferView is fully sized
    static constexpr bool is_sized_buffer = !contains_pointer && (true && ... && is_sized<Layout>);

    BufferView(glsl::StorageQualifier storageQualifier = glsl::StorageQualifier::std430)
        requires is_sized_buffer
    {
        constexpr std::size_t N = sizeof...(Layout);
        std::array<std::size_t, N> sizes = sizesOf<Layout...>(0, storageQualifier);
        std::array<std::size_t, N> alignments = alignmentsOf<Layout...>(storageQualifier);

        for (std::size_t i = 0; i < sizeof...(Layout); ++i) {
            fmt::println("{} -- {}", sizes[i], alignments[i]);
        }
    }

    BufferView(std::size_t size,
               glsl::StorageQualifier storageQualifier = glsl::StorageQualifier::std430)
        requires(!is_sized_buffer)
    {
        constexpr std::size_t N = sizeof...(Layout);
        std::array<std::size_t, N> sizes = sizesOf<Layout...>(size, storageQualifier);
        std::array<std::size_t, N> alignments = alignmentsOf<Layout...>(storageQualifier);

        for (std::size_t i = 0; i < sizeof...(Layout); ++i) {
            fmt::println("{} -- {}", sizes[i], alignments[i]);
        }
    }

  public:
    static std::size_t __size()
        requires(BufferView::is_sized_buffer)
    {
        return 0;
    }

    static std::size_t __size(std::size_t size)
        requires(!BufferView::is_sized_buffer)
    {
        return size;
    }

    static std::size_t __alignment() {
        return 0;
    }

  private:
    template <typename... Ts>
    std::array<std::size_t, sizeof...(Ts)> sizesOf(std::size_t c,
                                                   glsl::StorageQualifier storageQualifier) {
        return {subSize<Ts>(c, storageQualifier)...}; // Expands into {sizeof(T1), sizeof(T2), ...}
    }

    template <typename T>
    std::size_t subSize(std::size_t c, glsl::StorageQualifier storageQualifier) {
        if constexpr (glsl::primitive_like<T>) {
            return glsl::primitive_size<T>(storageQualifier);
        } else if constexpr (requires {
                                 { T::__size() } -> std::convertible_to<std::size_t>;
                             }) {
            return T::__size();
        } else if constexpr (requires(std::size_t c) {
                                 { T::__size(c) } -> std::convertible_to<std::size_t>;
                             }) {
            return T::__size(c);
        } else {
            static_assert(false);
        }
    }

    template <typename... Ts>
    std::array<std::size_t, sizeof...(Ts)> alignmentsOf(glsl::StorageQualifier storageQualifier) {
        return {
            subAlignmentsOf<Ts>(storageQualifier)...}; // Expands into {sizeof(T1), sizeof(T2), ...}
    }

    template <typename T> std::size_t subAlignmentsOf(glsl::StorageQualifier storageQualifier) {
        if constexpr (glsl::primitive_like<T>) {
            return glsl::primitive_size<T>(storageQualifier);
        } else if constexpr (requires {
                                 { T::__alignment() } -> std::convertible_to<std::size_t>;
                             }) {
            return T::__alignment();
        } else {
            static_assert(false);
            ;
        }
    }
};

inline void bar() {
    BufferView<"test1",                              //
               BufferView<"heavyCount", glsl::uint>, //
               BufferView<"heavyLight", float*>>     //
        test1{1};

    /* BufferView<"test2",                                   // */
    /*            BufferView<"counter", glsl::uint>,         // */
    /*            BufferView<"partitionStates",              // */
    /*                       BufferView<"agg", float>,       // */
    /*                       BufferView<"prefix", float>,    // */
    /*                       BufferView<"state", glsl::uint> // */
    /*                       >*                              // */
    /*            >                                          // */
    /*     test2{1}; */
    /* static_assert(decltype(test)::is_sized); */
}

}; // namespace wrs
