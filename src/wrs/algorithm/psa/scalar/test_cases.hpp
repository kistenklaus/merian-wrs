#pragma once
#include <src/wrs/gen/weight_generator.h>


namespace wrs::test::scalar_psa {

    struct TestCase {
        std::size_t weightCount;
        Distribution distribution;
        std::size_t splitCount;
        std::size_t iterations;
    };

    constexpr TestCase TEST_CASES[] = {
        TestCase {
            .weightCount = static_cast<std::size_t>(1024 * 2048),
            .distribution = Distribution::SEEDED_RANDOM_UNIFORM,
            .splitCount = static_cast<std::size_t>(1024 * 2048) / 32,
            .iterations = 1,
        },
    };

}
