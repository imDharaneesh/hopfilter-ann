#pragma once

#include <cstddef>
#include <cstdint>

#include <hopfilter/filter/selectivity.hpp>

namespace hopfilter {

struct AdaptiveThresholds {
    double postFilterAbove = 0.5;
    double exactBelow = 0.01;
};

struct FilteredSearchConfig {
    double overFetchFactor = 4.0;
    std::size_t rejectionWindow = 64;
    double rejectionThreshold = 0.9;
    double efGrowthFactor = 1.0;
    std::size_t maxEf = 1024;
    AdaptiveThresholds thresholds;
    std::size_t selectivitySample = kDefaultSelectivitySample;
    std::uint64_t selectivitySeed = kDefaultSelectivitySeed;
};

void validate(const FilteredSearchConfig& config);

}
