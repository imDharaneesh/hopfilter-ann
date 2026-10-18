#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/node_distances.hpp>

namespace hopfilter {

/// Strategy that picks which candidates become a node's neighbors.
///
/// ``candidates`` are sorted closest-first by distance to the base node, contain no
/// duplicates and exclude the base itself. Implementations are stateless and
/// thread-safe.
class NeighborSelector {
public:
    virtual ~NeighborSelector() = default;

    [[nodiscard]] virtual std::vector<NodeId> select(std::span<const Neighbor> candidates,
                                                     std::size_t maxNeighbors,
                                                     const NodeDistances& distances) const = 0;
};

/// Keeps the ``maxNeighbors`` closest candidates (the paper's Algorithm 3).
class SimpleNeighborSelector final : public NeighborSelector {
public:
    [[nodiscard]] std::vector<NodeId> select(std::span<const Neighbor> candidates,
                                             std::size_t maxNeighbors,
                                             const NodeDistances& distances) const override;
};

/// The paper's Algorithm 4: a candidate is kept only if it is closer to the base than
/// to every neighbor selected so far, which spreads edges in different directions and
/// keeps clustered data connected.
///
/// With ``keepPrunedConnections`` the remaining slots are filled with the closest
/// discarded candidates. The paper's ``extendCandidates`` option is not implemented;
/// it targets extremely clustered data and needs graph access during selection.
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
