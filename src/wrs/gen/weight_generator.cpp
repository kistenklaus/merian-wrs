#include "./weight_generator.h"

std::string wrs::distribution_to_pretty_string(Distribution dist) {
    switch (dist) {
    case Distribution::UNIFORM:
        return "uniform";
    case Distribution::PSEUDO_RANDOM_UNIFORM:
        return "pseudo_random_uniform";
    case Distribution::RANDOM_UNIFORM:
        return "random_uniform";
    case Distribution::SEEDED_RANDOM_UNIFORM:
        return "seeded_random_uniform";
    default:
        return "NO-PRETTY-STRING-AVAIL";
    }
}
