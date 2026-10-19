#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <hopfilter/common/types.hpp>
#include <hopfilter/distance/metrics.hpp>

namespace hopfilter {

inline constexpr std::size_t kDefaultMaxNeighbors = 16;
inline constexpr std::size_t kDefaultEfConstruction = 200;
inline constexpr std::uint64_t kDefaultSeed = 42;
inline constexpr std::size_t kLevelZeroDegreeFactor = 2;

struct IndexConfig {
    Dimension dimension = 0;
    std::size_t capacity = 0;
    std::size_t maxNeighbors = kDefaultMaxNeighbors;
    std::size_t efConstruction = kDefaultEfConstruction;
    std::string metricName{kL2MetricName};
    std::uint64_t seed = kDefaultSeed;

    [[nodiscard]] std::size_t maxNeighborsLevelZero() const noexcept {
        return kLevelZeroDegreeFactor * maxNeighbors;
    }
};

}
