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

/// Construction parameters of an HNSW index. Validated by HnswIndexBuilder.
struct IndexConfig {
    /// Length of every vector. Required.
    Dimension dimension = 0;
    /// Maximum number of vectors. Required; storage is allocated up front.
    std::size_t capacity = 0;
    /// M in the paper: neighbors per node on upper levels; level 0 allows twice as many.
    std::size_t maxNeighbors = kDefaultMaxNeighbors;
    /// Candidate list size while inserting; at least ``maxNeighbors``.
    std::size_t efConstruction = kDefaultEfConstruction;
    /// Registry name of the distance metric.
    std::string metricName{kL2MetricName};
    /// Seed of the level generator; the same seed and insert order give the same graph.
    std::uint64_t seed = kDefaultSeed;

    [[nodiscard]] std::size_t maxNeighborsLevelZero() const noexcept {
        return kLevelZeroDegreeFactor * maxNeighbors;
    }
};

}
