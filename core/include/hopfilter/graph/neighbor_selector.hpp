#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/node_distances.hpp>

namespace hopfilter {

class NeighborSelector {
public:
    virtual ~NeighborSelector() = default;

    [[nodiscard]] virtual std::vector<NodeId> select(std::span<const Neighbor> candidates,
                                                     std::size_t maxNeighbors,
                                                     const NodeDistances& distances) const = 0;
};

class SimpleNeighborSelector final : public NeighborSelector {
public:
    [[nodiscard]] std::vector<NodeId> select(std::span<const Neighbor> candidates,
                                             std::size_t maxNeighbors,
                                             const NodeDistances& distances) const override;
};

class HeuristicNeighborSelector final : public NeighborSelector {
public:
    explicit HeuristicNeighborSelector(bool keepPrunedConnections = false) noexcept
        : keepPrunedConnections_(keepPrunedConnections) {}

    [[nodiscard]] std::vector<NodeId> select(std::span<const Neighbor> candidates,
                                             std::size_t maxNeighbors,
                                             const NodeDistances& distances) const override;

private:
    bool keepPrunedConnections_;
};

}
