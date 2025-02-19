#include "./weight_generator.h"

std::string host::distribution_to_pretty_string(Distribution dist) {
    switch (dist) {
    case Distribution::UNIFORM:
        return "uniform";
    case Distribution::PSEUDO_RANDOM_UNIFORM:
        return "pseudo_random_uniform";
    case Distribution::RANDOM_UNIFORM:
        return "random_uniform";
    case Distribution::SEEDED_RANDOM_UNIFORM:
        return "seeded_random_uniform";
    case Distribution::SEEDED_RANDOM_EXPONENTIAL:
        return "seeded-random-exponential";
    case Distribution::SEEDED_RANDOM_NORMAL:
        return "seeded-random-normal";
    default:
        return "NO-PRETTY-STRING-AVAIL";
    }
}
