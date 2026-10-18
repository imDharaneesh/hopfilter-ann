#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace hopfilter {

using NodeId = std::uint32_t;
using LevelIndex = std::uint8_t;
using Dimension = std::size_t;

inline constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();

/// A stored node and its distance to some reference point; smaller is closer.
struct Neighbor {
    NodeId id;
    float distance;

    friend bool operator==(const Neighbor&, const Neighbor&) = default;
};

/// Strict weak ordering by ascending distance, ties broken by id so results are deterministic.
struct CloserFirst {
    [[nodiscard]] bool operator()(const Neighbor& lhs, const Neighbor& rhs) const noexcept {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.id < rhs.id;
    }
};

/// Neighbors sorted closest first, without duplicate ids.
using SearchResults = std::vector<Neighbor>;

}  // namespace hopfilter
