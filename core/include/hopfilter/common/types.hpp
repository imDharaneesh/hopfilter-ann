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

struct Neighbor {
    NodeId id;
    float distance;

    friend bool operator==(const Neighbor&, const Neighbor&) = default;
};

struct CloserFirst {
    [[nodiscard]] bool operator()(const Neighbor& lhs, const Neighbor& rhs) const noexcept {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.id < rhs.id;
    }
};

using SearchResults = std::vector<Neighbor>;

}
